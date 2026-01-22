class Solution:
    def knapsack(self, weights, values, capacity):
        # 0到i个物体任取放入容量为j背包时的最大价值
        dp = [[0 for _ in range(capacity + 1)] for _ in range(len(weights))]
        for i in range(len(weights)):
            dp[i][0] = 0
        for j in range(capacity, -1, -1):
            print(j)
            if j >= weights[0]:
                dp[0][j] = weights[0]
        print(dp)

        for i in range(len(weights)): #物品
            for j in range(capacity, weights[i]-1, -1):
                dp[i][j] = max(dp[i-1][j], dp[i-1][j-weights[i]] + values[i])

        print(dp)

        return dp[-1][-1]



# 示例测试
if __name__ == "__main__":
    solution = Solution()
    # 测试用例
    weights = [1, 2, 5, 6]
    values = [1, 6, 18, 22]
    capacity = 11
    max_value = solution.knapsack(weights, values, capacity)
    print(f"Maximum value in Knapsack = {max_value}")  # 输出: Maximum value in Knapsack = 40
