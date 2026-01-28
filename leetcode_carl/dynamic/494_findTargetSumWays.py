class Solution:
    def findTargetSumWays(self, nums: List[int], target: int) -> int:
        # dp[j] 装满这个容量为j背包有多少种方法
        weights = nums.copy()
        values = nums.copy()   
        sum_nums = sum(nums) 
        if abs(target) > sum_nums:
            return 0
        
        if (sum_nums + target) % 2 == 1:
            return 0

        capacity = (sum_nums + target) // 2
        dp = [0] * (capacity + 1)
        dp[0] = 1
        for i in range(len(weights)):
            for j in range(capacity, weights[i]-1, -1):
                dp[j] += dp[j-values[i]]

        return dp[-1]
