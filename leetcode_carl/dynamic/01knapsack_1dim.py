class Solution:
    def knapsack(self, weights, values, capacity):
        # 0到i个物体任取放入容量为j背包时的最大价值
        dp = [0] * (capacity + 1)
        
        for i in range(len(weights)):
            for j in range(capacity, weights[i]-1, -1):
                dp[j]  = max(dp[j], dp[j-weights[i]]+ values[i])

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
