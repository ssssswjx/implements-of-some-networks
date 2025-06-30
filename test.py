# from collections import deque
#
# # Function to check if a state is valid
# def is_valid_state(m_left, c_left, boat_left):
#     if m_left >= 0 and c_left >= 0 and m_left <= 3 and c_left <= 3:
#         m_right = 3 - m_left
#         c_right = 3 - c_left
#         if (m_left == 0 or m_left >= c_left) and (m_right == 0 or m_right >= c_right):
#             return True
#     return False
#
# # Function to perform BFS and find all solutions
# def bfs_missionaries_cannibals():
#     # Initial state: 3 missionaries and 3 cannibals on the left bank, boat on the left bank
#     start = (3, 3, 1)  # (missionaries_left, cannibals_left, boat_left)
#     goal = (0, 0, 0)   # Goal state: no one left on the left bank, boat on the right bank
#
#     # Queue for BFS: store current state and the path taken to reach it
#     queue = deque([(start, [])])
#
#     # Set to track visited states to avoid loops
#     visited = set([start])
#
#     # All possible moves
#     moves = [(1, 0), (2, 0), (1, 1), (0, 1), (0, 2)]  # (missionaries, cannibals)
#
#     solutions = []
#
#     while queue:
#         (m_left, c_left, boat_left), path = queue.popleft()
#
#         # If goal state is reached, store the solution
#         if (m_left, c_left, boat_left) == goal:
#             solutions.append(path + [(m_left, c_left, boat_left)])
#             continue
#
#         # Try all possible moves
#         for m_move, c_move in moves:
#             if boat_left == 1:  # Boat on the left bank
#                 new_state = (m_left - m_move, c_left - c_move, 0)
#             else:  # Boat on the right bank
#                 new_state = (m_left + m_move, c_left + c_move, 1)
#
#             if is_valid_state(*new_state) and new_state not in visited:
#                 visited.add(new_state)
#                 queue.append((new_state, path + [(m_left, c_left, boat_left)]))
#
#     return solutions
#
# # Main function to print all possible solutions
# def main():
#     solutions = bfs_missionaries_cannibals()
#     print(f"Found {len(solutions)} solutions.")
#     for i, solution in enumerate(solutions, 1):
#         print(f"\nSolution {i}:")
#         for state in solution:
#             print(f"Missionaries left: {state[0]}, Cannibals left: {state[1]}, Boat on left bank: {state[2] == 1}")
#
# if __name__ == "__main__":
#     main()
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches

# 创建一个新图
fig, ax = plt.subplots(figsize=(10, 8))

# 定义各个模块的位置和标签
modules = [
    ("输入噪声医学影像", (0.1, 0.8)),
    ("深度展开 + 加权L1范数", (0.5, 0.8)),
    ("WISTA-Net: 逐层展开收缩算法", (0.5, 0.6)),
    ("梯度下降 + 反向传播", (0.5, 0.4)),
    ("输出去噪医学影像", (0.9, 0.6)),
    ("自适应输入图像归一化（AIIN）", (0.5, 0.2)),
    ("生成高质量医学影像", (0.9, 0.2)),
]

# 绘制框和文本
for module, (x, y) in modules:
    ax.add_patch(mpatches.FancyBboxPatch((x-0.15, y-0.05), 0.3, 0.1, boxstyle="round,pad=0.1", edgecolor="black", facecolor="lightgray"))
    ax.text(x, y, module, ha="center", va="center", fontsize=10)

# 绘制箭头
arrows = [
    ((0.25, 0.8), (0.35, 0.8)),  # 输入到去噪算法
    ((0.65, 0.8), (0.65, 0.7)),  # 去噪算法到WISTA-Net
    ((0.65, 0.5), (0.65, 0.45)),  # WISTA-Net到梯度下降
    ((0.65, 0.4), (0.85, 0.4)),  # 梯度下降到输出去噪图像
    ((0.85, 0.6), (0.85, 0.55)),  # 输出去噪图像到归一化
    ((0.65, 0.2), (0.85, 0.2)),  # 自适应归一化到生成图像
]

for (start, end) in arrows:
    ax.annotate("", xy=end, xytext=start, arrowprops=dict(facecolor='black', shrink=0.05))

# 设置图形布局
ax.set_xlim(0, 1)
ax.set_ylim(0, 1)
ax.axis('off')  # 隐藏坐标轴

# 展示框图
plt.show()
