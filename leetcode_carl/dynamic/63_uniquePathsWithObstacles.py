class Solution:
    def uniquePathsWithObstacles(self, obstacleGrid):
        # s1. 确认dp数组含义 dp[i][j]到达第i行第j列有多少个路径
        # s2. 递推公式
        # s3. 初始化 
        # s4. 遍历顺序
        # s5. 打印数组

        m = len(obstacleGrid)
        n = len(obstacleGrid[0])
        dp = [[0 for _ in range(n) ] for _ in range(m)]
        if obstacleGrid[0][0] == 1 or obstacleGrid[m-1][n-1] == 1:
            return 0
        




        # 初始化
        for i in range(m):
            if obstacleGrid[i][0] == 1:
                dp[i][0] = 0
                break
            elif obstacleGrid[i][0] == 0:
                dp[i][0] = 1


        for j in range(n):
            if obstacleGrid[0][j] == 1:
                dp[0][j] = 0
                break
            elif obstacleGrid[0][j] == 0:
                dp[0][j] = 1
            

        for i in range(1,m):
            for j in range(1,n):
                if obstacleGrid[i][j] == 0:
                    dp[i][j] = dp[i-1][j] + dp[i][j-1]

        return dp[m-1][n-1]



# 示例测试
if __name__ == "__main__":
    solution = Solution()
    # 测试用例
    # grid1 = [
    #     [0, 0, 0],
    #     [0, 1, 0],
    #     [0, 0, 0],
    # ]
    # print(solution.uniquePathsWithObstacles(grid1))  # 输出: 2

    # grid2 = [
    #     [0, 1],
    #     [0, 0]
    # ]
    
    # print(solution.uniquePathsWithObstacles(grid2))  # 输出: 1

    grid3 = [[0,1,0,0]]
    print(solution.uniquePathsWithObstacles(grid3))  # 输出: 0
