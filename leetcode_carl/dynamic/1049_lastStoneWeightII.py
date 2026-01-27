class Solution:
    def lastStoneWeightII(self, stones):
        weights = stones.copy()
        values = stones.copy()
        sumStone = sum(stones)
        capacity = int(sum(stones) / 2  )
        # capacity = 1501
        dp = [0] * (capacity + 1)
        for i in range(len(weights)):
            for j in range(capacity, weights[i] - 1, -1):
                dp[j] = max(dp[j], dp[j - weights[i]] + values[i])
        

        return sumStone - dp[capacity] - dp[capacity]


# 示例测试
if __name__ == "__main__":
    solution = Solution()
    # 测试用例
    print(solution.lastStoneWeightII([2, 7, 4, 1, 8, 1]))  # 输出: 1
    print(solution.lastStoneWeightII([31, 26, 33, 21, 40]))  # 输出: 5
