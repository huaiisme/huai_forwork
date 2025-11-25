class Solution:
    def sortedSquares(self, nums: List[int]) -> List[int]:
        record = []
        len_nums = len(nums)
        left = 0
        right = len_nums -1
        while left <= right:
            if nums[left] ** 2 > nums[right] ** 2:
                record.append(nums[left]** 2)
                left += 1
                # print(record)
            elif nums[left] ** 2 <= nums[right] ** 2:
                record.append(nums[right]** 2)
                right -= 1
        record.reverse()
        return record
        