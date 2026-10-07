using System.Diagnostics;
using Unity.AI.Navigation.LowLevel;
using Unity.Collections;
using Unity.Jobs;
using UnityEngine;

/// <summary>
/// 批量可达性查询与路径拐点提取 Job
/// 在单个 IJob 中串行处理多组查询，适用于需要精确控制单线程工作负载的场景
/// 核心能力：判定连通性 + 提取简化路径拐点（Portal 中点）
/// </summary>
public struct ReachabilityJob : IJob
{
    // ===== 导航世界与查询缓冲区 =====
    // NavWorld 是 Unity 6.6 新导航系统的核心句柄，线程安全且支持并发读取
    public NavWorld world;
    // NavQueryBuffer 是单次查询的临时状态容器，每个 Job 实例独占一个以避免竞争
    public NavQueryBuffer buffer;

    // ===== 输入数据（只读）=====
    [ReadOnly] public NativeArray<Vector3> from;   // 批量查询的起点集合
    [ReadOnly] public NativeArray<Vector3> to;     // 批量查询的终点集合（与 from 一一对应）

    // ===== 输出数据 =====
    public NativeArray<byte> reachable;      // 每对点的可达性结果：1=可达，0=不可达
    public NativeArray<float> workerMs;      // 整个 Job 的执行耗时（毫秒），用于性能监控
    public NativeArray<int> cornerCounts;    // 每条路径的实际拐点数量
    public NativeArray<Vector3> corners;     // 扁平化的拐点数组，第 i 条路径的数据位于 [i*maxCorners, (i+1)*maxCorners)

    // ===== 查询控制参数 =====
    public int nodesToVisit;   // ContinueFindPath 每次迭代最多访问的节点数，控制单次步进的计算预算
    public int maxCorners;     // 每条路径允许存储的最大拐点数，超出部分将被截断

    /// <summary>
    /// Job 执行入口：串行遍历所有查询对，依次执行寻路并提取结果
    /// </summary>
    public void Execute()
    {
        var sw = Stopwatch.StartNew();

        for (int i = 0; i < from.Length; i++)
        {
            // 初始化当前查询的拐点计数
            cornerCounts[i] = 0;

            // 【坐标映射】将世界空间坐标转换为 NavWorld 内部的导航位置句柄
            // Vector3.one * 2f 为搜索半径容差，0 为代理类型索引
            var start = world.MapLocation(from[i], Vector3.one * 2f, 0);
            var end = world.MapLocation(to[i], Vector3.one * 2f, 0);

            // 若起点或终点无法映射到有效导航区域，直接标记为不可达并跳过
            if (!world.IsValid(start) || !world.IsValid(end))
            {
                reachable[i] = 0;
                continue;
            }

            // 【非阻塞寻路 - 启动阶段】
            // BeginFindPath 初始化查询状态到 buffer 中，返回初始状态标志
            var status = world.BeginFindPath(buffer, start, end);

            // 【非阻塞寻路 - 步进阶段】
            // 循环调用 ContinueFindPath 逐步推进搜索，直到完成或达到安全上限
            // guard 防止极端情况下无限循环（如超大地图或损坏的导航网格）
            int guard = 0;
            while ((status & NavQueryStatus.InProgress) != 0 && guard++ < 512)
                status = world.ContinueFindPath(buffer, nodesToVisit, out _);

            // 若最终状态不包含 Success 标志，标记为不可达
            if ((status & NavQueryStatus.Success) == 0)
            {
                reachable[i] = 0;
                continue;
            }

            // 【非阻塞寻路 - 结束阶段】
            // EndFindPath 终结查询并获取原始路径节点数量
            status = world.EndFindPath(buffer, out int pathSize);

            // 严格判定路径完整性：成功 + 非部分结果 + 至少有一个节点
            bool complete = (status & NavQueryStatus.Success) != 0
                            && (status & NavQueryStatus.PartialResult) == 0
                            && pathSize > 0;
            reachable[i] = (byte)(complete ? 1 : 0);

            if (!complete)
                continue;

            // 【路径提取】从 buffer 中拷贝原始 NavNode 序列到临时数组
            var path = new NativeArray<NavNode>(pathSize, Allocator.Temp);
            int copied = world.GetResultFromFindPath(buffer, path);

            // 【拐点生成】将 NavNode 路径简化为可渲染/可跟随的拐点序列
            int count = 0;
            int offset = i * maxCorners; // 计算当前路径在扁平化 corners 数组中的起始偏移

            // 第一个拐点始终为起点
            corners[offset + count++] = start.position;

            // 遍历相邻 NavNode 对，通过 Portal 中点生成中间拐点
            // gates 限制中间拐点数不超过 maxCorners - 2（预留起点和终点的位置）
            int gates = Mathf.Min(copied - 1, maxCorners - 2);
            for (int n = 0; n < gates; n++)
            {
                // GetPortalPoints 获取两个相邻导航节点之间连接门廊的左右边界点
                // 取中点作为简化路径的拐点，这是 Funnel Algorithm 的前置步骤
                if (!world.GetPortalPoints(path[n], path[n + 1], out Vector3 left, out Vector3 right))
                    continue;
                corners[offset + count++] = (left + right) * 0.5f;
            }

            // 最后一个拐点为终点（仅在数组容量允许时写入）
            if (count < maxCorners)
                corners[offset + count++] = end.position;

            cornerCounts[i] = count;

            // Temp 分配必须在同帧内释放，避免内存泄漏
            path.Dispose();
        }

        sw.Stop();
        // 记录整个批次查询的总耗时，供主线程读取以评估性能
        workerMs[0] = (float)sw.Elapsed.TotalMilliseconds;
    }
}