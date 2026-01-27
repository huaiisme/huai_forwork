class Solution:
    def canPartition(self, nums):
        capacity = sum(nums)
        if capacity % 2 == 1:
            return False
        capacity //= 2
        weights = nums.copy()
        values = nums.copy()
        dp = [0] * (capacity + 1)
        for i in range(len(weights)):
            for j in range(capacity, weights[i]-1, -1):
                dp[j] = max(dp[j], dp[j- weights[i]] + values[i])

        return capacity == dp[capacity]

# 示例测试
if __name__ == "__main__":
    solution = Solution()
    # 测试用例
    print(solution.canPartition([1, 5, 11, 5]))  # 输出: True
    print(solution.canPartition([1, 2, 3, 5]))    # 输出: False
