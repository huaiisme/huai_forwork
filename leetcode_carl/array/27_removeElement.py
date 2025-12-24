# 27 removeElement
# debug part

nums = [3,2,2,3] # final return:2, nums = [2,2,_,_]
val = 3

slow = 0 
fast = 0

len_nums = len(nums)

for fast in range(len_nums):
    if nums[fast] != val:
        nums[slow] = nums[fast]
        slow += 1

print(nums[:slow])

# leetcode actual part
# class Solution:
#     def removeElement(self, nums: List[int], val: int) -> int:
#         slow = 0 
#         fast = 0

#         len_nums = len(nums)

#         for fast in range(len_nums):
#             if nums[fast] != val:
#                 nums[slow] = nums[fast]
#                 slow += 1

#         return len(nums[:slow])
        




