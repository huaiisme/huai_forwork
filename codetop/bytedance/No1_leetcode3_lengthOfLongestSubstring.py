# 严格符合力扣官方函数定义：函数名、参数、返回值完全一致
class Solution:
    def __init__(self):
        self.maxlength = 0

    def lengthOfLongestSubstring(self, s: str) -> int:
        len_s = len(s)
        slow = 0
        fast = 0
        if len_s == 0:
            return 0
        if len_s == 1:
            return 1

        for slow in range(len_s):
            theDict = dict()
            for fast in range(slow, len_s):
                if s[fast] not in theDict:
                    theDict[s[fast]] = 1
                else:
                    if len(theDict) > self.maxlength:
                        self.maxlength = len(theDict)
                    break
                    
                if fast == len_s - 1:
                    if len(theDict) > self.maxlength:
                        self.maxlength = len(theDict)

        return self.maxlength

# 本地调试入口：独立运行时执行，提交力扣时可注释/保留（不影响提交）
if __name__ == "__main__":
    # 定义测试用例：覆盖常规情况、边界情况、特殊情况
    test_cases = [
        ("abcabcbb", 3),   # 常规重复，最长为abc/bca/cab
        ("bbbbb", 1),      # 全重复字符
        ("pwwkew", 3),     # 中间重复，两端有效
        ("", 0),            # 空字符串边界
        ("au", 2),          # 无重复短字符串
        ("dvdf", 3)         # 非连续重复特殊情况
    ]
    
    # 遍历测试用例并验证结果
    for idx, (input_s, expected) in enumerate(test_cases, 1):
        result = Solution().lengthOfLongestSubstring(input_s)
        status = "✅ 成功" if result == expected else "❌ 失败"
        print(f"测试用例 {idx}：{status}")
        print(f"  输入：{input_s!r}")
        print(f"  预期：{expected}，实际：{result}\n")
