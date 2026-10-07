using Unity.AI.Navigation.LowLevel;
using Unity.Collections;
using Unity.Jobs;
using UnityEngine;
using UnityEngine.AI;
using UnityEngine.InputSystem;
using UnityEngine.Profiling;
using UnityEngine.Rendering;

/// <summary>
/// NavWorld 多线程寻路性能基准测试器
/// 支持在 Worker 线程（NavWorld + Job）与主线程（传统 NavMesh API）之间实时切换对比
/// 通过 GPU 友好的 Mesh 线条可视化查询结果，避免 Gizmo/Debug.DrawLine 的性能干扰
/// </summary>
public class NavWorldBenchmark : MonoBehaviour
{
    #region Fields

    // ===== 可调参数 =====
    [SerializeField] int agentCount = 200;           // 批量查询的对数
    [SerializeField] bool useWorkerThread = true;    // true=Worker线程, false=主线程
    [SerializeField] float worldHalfExtent = 32f;    // 随机采样点的半范围
    [SerializeField] int nodesToVisit = 64;          // Job 中 ContinueFindPath 每步节点预算
    [SerializeField] int maxCorners = 32;            // 每条路径最大拐点数
    [SerializeField] Material queryLinesMaterial;    // 可选：自定义线条材质
    [SerializeField] GameObject queryLinesPrefab;    // 可选：自定义线条预制体

    // ===== Native 数据容器（Persistent 生命周期跨帧复用）=====
    NativeArray<Vector3> from;         // 起点数组
    NativeArray<Vector3> to;           // 终点数组
    NativeArray<byte> reachable;       // Job 输出的可达性标记
    NativeArray<float> workerMs;       // Job 输出的执行耗时
    NativeArray<int> cornerCounts;     // Job 输出的每条路径拐点数
    NativeArray<Vector3> corners;      // Job 输出的扁平化拐点数据
    byte[] snapshot;                   // 主线程可读的可达性快照（用于渲染和HUD）

    // ===== 导航系统句柄 =====
    NavWorld world;                    // Unity 6.6 新导航世界
    NavQueryBuffer buffer;             // 查询状态缓冲区
    NavMeshPath path;                  // 传统 API 复用的路径对象

    // ===== Job 调度状态 =====
    JobHandle handle;                  // 当前异步 Job 句柄
    bool scheduled;                    // 是否有 Job 正在执行中

    // ===== 性能指标缓存 =====
    float lastMainMs;                  // 上次主线程查询耗时
    float lastWorkerMs;                // 上次 Worker 线程查询耗时
    int lastReachableCount;            // 上次成功连通的路径数

    // ===== 可视化相关 =====
    GUIStyle hudStyle;                 // HUD 文字样式缓存
    GameObject lineObject;             // 线条 Mesh 载体
    Mesh lineMesh;                     // 动态生成的线段 Mesh
    Material lineMaterial;             // 线条渲染材质
    bool ownsLineMaterial;             // 是否由本组件创建材质（决定销毁权）
    Vector3[] lineVertices;            // CPU 端顶点缓冲
    Color[] lineColors;                // CPU 端颜色缓冲
    int linesChecksum;                 // 上一帧的可视化哈希，避免无变化时重建 Mesh

    // ===== 公共只读属性（供外部 UI 或测试框架读取）=====
    public float LastMainMs => lastMainMs;
    public float LastWorkerMs => lastWorkerMs;
    public int LastReachableCount => lastReachableCount;
    public bool UseWorkerThread => useWorkerThread;
    public bool JobInFlight => scheduled;

    #endregion

    // OnEnable 和 Start 双重保障，确保在任意激活时机都完成初始化
    private void OnEnable() => EnsureAllocated();
    private void Start() => EnsureAllocated();

    /// <summary>
    /// 延迟分配所有 Native 容器和导航资源
    /// 使用 IsCreated 检查实现幂等初始化，避免重复分配
    /// </summary>
    private void EnsureAllocated()
    {
        if (from.IsCreated) return;

        // 分配 Persistent 容器：生命周期与组件绑定，跨帧复用避免 GC
        from = new NativeArray<Vector3>(agentCount, Allocator.Persistent);
        to = new NativeArray<Vector3>(agentCount, Allocator.Persistent);
        reachable = new NativeArray<byte>(agentCount, Allocator.Persistent);
        workerMs = new NativeArray<float>(1, Allocator.Persistent);
        cornerCounts = new NativeArray<int>(agentCount, Allocator.Persistent);
        corners = new NativeArray<Vector3>(agentCount * maxCorners, Allocator.Persistent);
        snapshot = new byte[agentCount];

        // 获取默认导航世界并创建查询缓冲区
        world = NavWorld.GetDefaultWorld();
        buffer = new NavQueryBuffer(world, Allocator.Persistent, 2000);

        // 传统 API 复用的路径对象（避免每帧 new）
        path = new NavMeshPath();

        ScramblePairs();
    }

    /// <summary>
    /// 安全释放所有资源，处理未完成的 Job 防止内存泄漏
    /// </summary>
    private void OnDisable()
    {
        if (!from.IsCreated) return;

        // ⚠️ 关键：禁用前必须等待进行中的 Job 完成
        // 否则 Native 容器被 Dispose 后 Job 仍在写入会导致崩溃
        if (scheduled)
        {
            handle.Complete();
            scheduled = false;
            ApplyWorkerResults();
        }

        buffer.Dispose();
        world.Dispose();
        ReleaseLines();

        // 逐一安全释放 Native 容器
        if (from.IsCreated) from.Dispose();
        if (to.IsCreated) to.Dispose();
        if (reachable.IsCreated) reachable.Dispose();
        if (workerMs.IsCreated) workerMs.Dispose();
        if (cornerCounts.IsCreated) cornerCounts.Dispose();
        if (corners.IsCreated) corners.Dispose();
    }

    /// <summary>
    /// 每帧更新：处理输入并根据模式分发到主线程或 Worker 线程
    /// </summary>
    private void Update()
    {
        if (!from.IsCreated) return;

        // Input System 快捷键
        var keyboard = Keyboard.current;
        if (keyboard != null)
        {
            if (keyboard.spaceKey.wasPressedThisFrame) ScramblePairs();   // Space: 重新随机起终点对
            if (keyboard.tabKey.wasPressedThisFrame) useWorkerThread = !useWorkerThread; // Tab: 切换模式
        }

        // 根据当前模式执行对应的查询逻辑
        if (useWorkerThread) TickWorker();
        else TickMainThread();
    }

    /// <summary>
    /// LateUpdate 中重建可视化线条
    /// 放在 LateUpdate 确保在所有查询结果就绪后再渲染
    /// </summary>
    private void LateUpdate()
    {
        if (snapshot == null || !from.IsCreated) return;

        EnsureLines();

        // 通过校验和判断数据是否变化，避免每帧无条件重建 Mesh
        int hash = PictureChecksum();
        if (hash == linesChecksum) return;

        linesChecksum = hash;
        RebuildLines();
    }

    /// <summary>
    /// 主线程同步查询模式
    /// 使用传统 NavMesh.CalculatePath 作为性能基线
    /// </summary>
    private void TickMainThread()
    {
        // 若之前有 Worker Job 未完成，先回收其结果再切换到主线程模式
        if (scheduled)
        {
            handle.Complete();
            scheduled = false;
            ApplyWorkerResults();
        }

        int hits = 0;
        Profiler.BeginSample("NavMesh.MainThread.200Queries");
        float t0 = Time.realtimeSinceStartup;

        for (int i = 0; i < agentCount; i++)
        {
            // 传统同步 API：阻塞主线程直到查询完成
            bool ok = NavMesh.CalculatePath(from[i], to[i], NavMesh.AllAreas, path);
            bool complete = ok && path.status == NavMeshPathStatus.PathComplete;
            snapshot[i] = (byte)(complete ? 1 : 0);
            if (complete) hits++;
        }

        lastMainMs = (Time.realtimeSinceStartup - t0) * 1000f;
        Profiler.EndSample();
        lastReachableCount = hits;
        lastWorkerMs = 0f; // 主线程模式下 Worker 耗时归零
    }

    /// <summary>
    /// Worker 线程异步查询模式
    /// 采用非阻塞调度：每帧检查上一帧 Job 是否完成，完成后立即调度下一批
    /// 实现"流水线"效果，主线程几乎零开销
    /// </summary>
    private void TickWorker()
    {
        // 检查上一帧调度的 Job 是否已完成
        if (scheduled)
        {
            if (!handle.IsCompleted) return; // 未完成则跳过本帧，不阻塞主线程

            handle.Complete();
            scheduled = false;

            // 读取 Worker 输出的耗时和可达性结果
            lastWorkerMs = workerMs[0];
            reachable.CopyTo(snapshot);

            int hits = 0;
            for (int i = 0; i < agentCount; i++)
                if (snapshot[i] != 0) hits++;
            lastReachableCount = hits;
        }

        // 调度新的 ReachabilityJob
        var job = new ReachabilityJob
        {
            world = world,
            buffer = buffer,
            from = from,
            to = to,
            reachable = reachable,
            workerMs = workerMs,
            cornerCounts = cornerCounts,
            corners = corners,
            nodesToVisit = nodesToVisit,
            maxCorners = maxCorners
        };

        Profiler.BeginSample("NavWorld.Schedule.WorkerQueries");
        handle = job.Schedule();

        // ⚠️ 关键：将 Job 依赖注册到 NavWorld
        // 确保 NavWorld 的内部更新不会与本 Job 产生数据竞争
        world.AddDependency(handle);

        Profiler.EndSample();
        scheduled = true;
        lastMainMs = 0f; // Worker 模式下主线程耗时归零
    }

    /// <summary>
    /// 将 Worker 输出拷贝到主线程可读的 snapshot 并统计结果
    /// 在 Job 完成和模式切换时调用
    /// </summary>
    private void ApplyWorkerResults()
    {
        if (!reachable.IsCreated || snapshot == null) return;

        reachable.CopyTo(snapshot);

        int hits = 0;
        for (int i = 0; i < snapshot.Length; i++)
            if (snapshot[i] != 0) hits++;

        lastReachableCount = hits;
        lastWorkerMs = workerMs.IsCreated ? workerMs[0] : 0f;
    }

    /// <summary>
    /// 随机生成新的起终点对
    /// 调度前强制等待进行中的 Job 完成，防止写入正在被 Job 读取的数组
    /// </summary>
    private void ScramblePairs()
    {
        if (scheduled)
        {
            handle.Complete();
            scheduled = false;
        }

        for (int i = 0; i < agentCount; i++)
        {
            from[i] = RandomNavPoint();
            to[i] = RandomNavPoint();
            snapshot[i] = 0;
        }
    }

    /// <summary>
    /// 在世界范围内采样一个有效的导航网格位置
    /// 若 SamplePosition 失败则返回原始随机点（保证不会因采样失败而中断测试）
    /// </summary>
    private Vector3 RandomNavPoint()
    {
        var p = new Vector3(
            Random.Range(-worldHalfExtent, worldHalfExtent),
            0f,
            Random.Range(-worldHalfExtent, worldHalfExtent));

        return NavMesh.SamplePosition(p, out NavMeshHit hit, 2f, NavMesh.AllAreas)
            ? hit.position
            : p;
    }

    /// <summary>
    /// IMGUI HUD 面板：实时显示模式、耗时、连通率等关键指标
    /// 使用 GUI.matrix 缩放适配不同分辨率
    /// </summary>
    private void OnGUI()
    {
        float scale = Mathf.Max(1.8f, Screen.height / 520f);
        var previousMatrix = GUI.matrix;
        GUI.matrix = Matrix4x4.TRS(Vector3.zero, Quaternion.identity, Vector3.one * scale);

        hudStyle ??= new GUIStyle(GUI.skin.label)
        {
            fontSize = 20,
            fontStyle = FontStyle.Bold,
            wordWrap = true,
            padding = new RectOffset(22, 22, 16, 16)
        };
        hudStyle.normal.textColor = Color.white;

        var panel = new Rect(18, 18, 860, 100);

        // 半透明深色背景
        var previousColor = GUI.color;
        GUI.color = new Color(0.04f, 0.05f, 0.07f, 0.9f);
        GUI.DrawTexture(panel, Texture2D.whiteTexture);
        GUI.color = previousColor;

        string mode = useWorkerThread ? "WORKER" : "MAIN";
        GUI.Label(panel,
            $"{mode}     {lastReachableCount} / {agentCount} complete\n" +
            $"Main {lastMainMs:0.00} ms      Worker {lastWorkerMs:0.00} ms\n" +
            "No agents. Each line is one question.\n" +
            "Green means a complete path exists.\n" +
            "The rest have one end in the closed room.\n" +
            "Tab = mode      Space = new pairs",
            hudStyle);

        GUI.matrix = previousMatrix;
    }

    /// <summary>
    /// 懒初始化线条 Mesh 及其渲染组件
    /// 使用 Quad 条带代替 LineRenderer/Gizmo，获得更好的批处理和 LOD 控制
    /// </summary>
    private void EnsureLines()
    {
        if (lineMesh != null) return;

        lineMesh = new Mesh { name = "QueryLines" };
        lineMesh.MarkDynamic(); // 提示 Unity 此 Mesh 频繁更新，优化上传策略

        // 每条查询线 = 4 顶点 + 2 三角形（一个 Quad）
        lineVertices = new Vector3[agentCount * 4];
        lineColors = new Color[agentCount * 4];
        var triangles = new int[agentCount * 6];

        for (int i = 0; i < agentCount; i++)
        {
            int v = i * 4;
            int t = i * 6;
            // 两个三角形组成一个 Quad
            triangles[t]     = v;
            triangles[t + 1] = v + 1;
            triangles[t + 2] = v + 2;
            triangles[t + 3] = v;
            triangles[t + 4] = v + 2;
            triangles[t + 5] = v + 3;
        }

        lineMesh.vertices = lineVertices;
        lineMesh.colors = lineColors;
        lineMesh.triangles = triangles;

        // 创建或使用预制体作为 Mesh 载体
        if (queryLinesPrefab != null)
            lineObject = Instantiate(queryLinesPrefab, transform, false);
        else
            lineObject = new GameObject("QueryLines");

        lineObject.name = "QueryLines";
        lineObject.transform.SetParent(transform, false);

        var filter = lineObject.GetComponent<MeshFilter>();
        if (filter == null) filter = lineObject.AddComponent<MeshFilter>();
        filter.sharedMesh = lineMesh;

        var renderer = lineObject.GetComponent<MeshRenderer>();
        if (renderer == null) renderer = lineObject.AddComponent<MeshRenderer>();

        lineMaterial = ResolveQueryLinesMaterial();
        if (lineMaterial == null) return;

        renderer.sharedMaterial = lineMaterial;
        renderer.shadowCastingMode = ShadowCastingMode.Off; // 线条不参与阴影
        renderer.receiveShadows = false;
    }

    /// <summary>
    /// 解析线条材质：优先使用指定材质 → 专用 Shader → URP Unlit 兜底
    /// ownsLineMaterial 标记决定是否由本组件负责销毁
    /// </summary>
    private Material ResolveQueryLinesMaterial()
    {
        ownsLineMaterial = false;

        if (queryLinesMaterial != null)
            return queryLinesMaterial;

        var shader = Shader.Find("NavWorld/QueryLines");
        if (shader == null)
            shader = Shader.Find("Universal Render Pipeline/Unlit");

        if (shader == null)
        {
            Debug.LogError("Missing shader 'NavWorld/QueryLines'. Assign QueryLines.mat on NavWorldBenchmark.", this);
            return null;
        }

        ownsLineMaterial = true;
        return new Material(shader);
    }

    /// <summary>
    /// 根据最新 snapshot 重建线条 Mesh 的顶点和颜色
    /// 绿色=可达，红色=不可达；不可达线条略宽以增强视觉区分
    /// </summary>
    private void RebuildLines()
    {
        const float height = 0.2f;       // 线条离地高度，避免 Z-fighting
        const float halfWidth = 0.045f;  // 线条半宽
        var reached = new Color(0.15f, 0.95f, 0.3f);  // 可达：亮绿
        var blocked = new Color(1f, 0.2f, 0.12f);     // 不可达：橙红

        for (int i = 0; i < agentCount; i++)
        {
            bool open = snapshot[i] == 0;
            float width = open ? halfWidth * 2.2f : halfWidth; // 不可达线条加宽

            var a = from[i];
            var b = to[i];
            a.y = open ? height + 0.12f : height; // 不可达线条略高，分层显示
            b.y = a.y;

            // 计算垂直于连线方向的侧向偏移，构建 Quad 四个角
            var dir = b - a;
            dir.y = 0f;
            var side = dir.sqrMagnitude < 0.0001f
                ? Vector3.right * width
                : Vector3.Cross(Vector3.up, dir.normalized) * width;

            int v = i * 4;
            lineVertices[v]     = a - side;
            lineVertices[v + 1] = a + side;
            lineVertices[v + 2] = b + side;
            lineVertices[v + 3] = b - side;

            var color = snapshot[i] != 0 ? reached : blocked;
            lineColors[v] = lineColors[v + 1] = lineColors[v + 2] = lineColors[v + 3] = color;
        }

        lineMesh.vertices = lineVertices;
        lineMesh.colors = lineColors;
        lineMesh.RecalculateBounds(); // 更新包围盒以确保正确的剔除和排序
    }

    /// <summary>
    /// 计算当前可视化状态的校验和
    /// 仅当 snapshot 内容或首尾坐标变化时才触发 RebuildLines
    /// 避免数据未变时每帧上传 Mesh 造成的 GPU 带宽浪费
    /// </summary>
    private int PictureChecksum()
    {
        int hash = agentCount;
        for (int i = 0; i < agentCount; i++)
            hash = unchecked(hash * 31 + snapshot[i]);

        if (agentCount > 0)
            hash = unchecked(hash + from[0].GetHashCode() + to[agentCount - 1].GetHashCode());

        return hash;
    }

    /// <summary>
    /// 安全释放线条相关的所有资源
    /// 区分 Play Mode 和 Editor Mode 使用不同的销毁 API
    /// </summary>
    private void ReleaseLines()
    {
        linesChecksum = 0;

        if (lineObject != null)
        {
            if (Application.isPlaying) Destroy(lineObject);
            else DestroyImmediate(lineObject);
            lineObject = null;
        }

        if (lineMesh != null)
        {
            if (Application.isPlaying) Destroy(lineMesh);
            else DestroyImmediate(lineMesh);
            lineMesh = null;
        }

        // 仅销毁由本组件创建的材质，避免误删用户指定的共享材质
        if (ownsLineMaterial && lineMaterial != null)
        {
            if (Application.isPlaying) Destroy(lineMaterial);
            else DestroyImmediate(lineMaterial);
        }

        lineMaterial = null;
        ownsLineMaterial = false;
    }
}