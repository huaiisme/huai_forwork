class Solution:
    def uniquePathsWithObstacles(self, obstacleGrid: list[list[int]]) -> int:
    

# 示例测试
if __name__ == "__main__":
    solution = Solution()
    # 测试用例
    grid1 = [
        [0, 0, 0],
        [0, 1, 0],
        [0, 0, 0]
    ]
    print(solution.uniquePathsWithObstacles(grid1))  # 输出: 2

    grid2 = [
        [0, 1],
        [0, 0]
    ]
    print(solution.uniquePathsWithObstacles(grid2))  # 输出: 1
