## [7. 整数反转](https://leetcode.cn/problems/reverse-integer/)

<img src="./%E6%95%B0%E5%AD%A6-%E5%8D%9A%E5%BC%88-%E8%AE%A1%E7%BB%84-OS%E7%AD%89.assets/image-20230714225347846.png" alt="image-20230714225347846" style="zoom:67%;" />

> 难点就在于不能使用超过32位的数字
>
> rev在x还有数字的时候, 下一步肯定是要*10, 再加一个数
>
> 如果比Max/Min的1/10大/小, *10之后一定不满足

```go
func reverse(x int) int {
	rev := 0
	for x != 0 {
		if rev < math.MinInt32/10 || rev > math.MaxInt32/10 {
			return 0
		}
		rev = rev*10 + x%10
		x /= 10
	}
	return rev
}
```

## [8. 字符串转换整数 (atoi)](https://leetcode.cn/problems/string-to-integer-atoi/)

<img src="./%E6%95%B0%E5%AD%A6-%E5%8D%9A%E5%BC%88-%E8%AE%A1%E7%BB%84-OS%E7%AD%89.assets/image-20230714234919983.png" alt="image-20230714234919983" style="zoom:67%;" />

> - 多次判断长度是否为0
> - 再循环过程中判断终止条件, 要不然可能会超出Int的值

```go
func myAtoi(s string) int {
	if len(s) == 0 {
		return 0
	}
	for len(s) > 0 && s[0] == ' ' {
		s = s[1:]
	}

	if len(s) == 0 {
		return 0
	}

	rev, flag := 0, 1

	if s[0] == '+' || s[0] == '-' {
		if s[0] == '-' {
			flag = -flag
		}
		s = s[1:]
	}

	fmt.Println(flag)
	for _, item := range s {
		if !(item >= '0' && item <= '9') {
			return rev * flag
		}
		rev = rev*10 + int(item) - int('0')

		if flag < 0 && rev*flag < math.MinInt32 {
			return math.MinInt32
		}
		if flag > 0 && rev > math.MaxInt32 {
			return math.MaxInt32
		}
	}

	return rev * flag
}
```





## [9. 回文数](https://leetcode.cn/problems/palindrome-number/)

<img src="./%E6%95%B0%E5%AD%A6-%E5%8D%9A%E5%BC%88-%E8%AE%A1%E7%BB%84-OS%E7%AD%89.assets/image-20230714225702487.png" alt="image-20230714225702487" style="zoom:67%;" />

> 回文数一定不会溢出的
>
> 所以溢出判断不相等就ok了啊
>
> 也是可以只反转一半的哈哈.....

```go
func isPalindrome(x int) bool {
	if x < 0 {
		return false
	}
	tmpX := x
	rev := 0
	for tmpX != 0 {
		rev = rev*10 + tmpX%10
		tmpX /= 10
	}
	return x == rev
}
```

## [29. 两数相除](https://leetcode.cn/problems/divide-two-integers/)

<img src="./%E6%95%B0%E5%AD%A6-%E5%8D%9A%E5%BC%88-%E8%AE%A1%E7%BB%84-OS%E7%AD%89.assets/image-20230717152652943.png" alt="image-20230717152652943" style="zoom: 67%;" />

> 负数比正数的范围更大, 这里千万注意数据范围最小是-2147483638, 因此如果转换为正数, 会很悲惨的溢出, 所以需要转换为负数.
>
> 对于一次次减是不可行的, 但咱们可以减除数的2倍,然后结果+2,4倍+4...... 故不停的左移除数, 直到其大于被除数的一半, 然后减去, 右移除数使其小于被除数,减去......依次类推, 直到被除数小于原始除数.
>
> 但是做减法, 时间太久了!!!!
>
> 
>
> 倍乘, 倍除的思想

```go
func divide(dividend int, divisor int) int {
	// 异常例子
	if dividend == math.MinInt32 && divisor == -1 {
		return math.MaxInt32
	}
	if divisor == 0 {
		return 0
	}

	flag, res := 1, 0
	// 都转为 < 0 小于0的范围更大
	if dividend > 0 {
		flag *= -1
		dividend *= -1
	}
	if divisor > 0 {
		flag *= -1
		divisor *= -1
	}

	now, tmpDivisor := 1, divisor
	// 现在tmpDivisor是比dividend大于等于了
	for tmpDivisor >= dividend {
		now <<= 1
		tmpDivisor <<= 1
	}

	for dividend <= divisor {
		// 从大向小
		for tmpDivisor < dividend {
			now >>= 1
			tmpDivisor >>= 1
		}
		res += now
		dividend -= tmpDivisor
	}
	return res * flag
}
```

## [38. 外观数列](https://leetcode.cn/problems/count-and-say/)

<img src="./%E6%95%B0%E5%AD%A6-%E5%8D%9A%E5%BC%88-%E8%AE%A1%E7%BB%84-OS%E7%AD%89.assets/image-20230718101619350.png" alt="image-20230718101619350" style="zoom:67%;" />

> 遍历和递归的思想都是一样的哈
>
> 从小数到大数一直走就行了
>
> 不要怕, 找到规律开干就ok
>
> 
>
> 要进行收尾工作的哈
>
> 数字转字符串使用strconv

```go
func countAndSay(n int) string {
	if n == 1 {
		return "1"
	}
	tmpN := countAndSay(n - 1)
	var res bytes.Buffer
	cnt := 1
	num := tmpN[0]
	for i := 1; i < len(tmpN); i++ {
		if tmpN[i] == num {
			cnt += 1
		} else {
			res.WriteString(strconv.Itoa(cnt))
			res.WriteByte(num)
			num = tmpN[i]
			cnt = 1
		}
	}
	// 收尾工作
	res.WriteString(strconv.Itoa(cnt))
	res.WriteByte(num)

	return res.String()
}
```

## [50. Pow(x, n)](https://leetcode.cn/problems/powx-n/)

<img src="./%E6%95%B0%E5%AD%A6-%E5%8D%9A%E5%BC%88-%E8%AE%A1%E7%BB%84-OS%E7%AD%89.assets/image-20230718185106666.png" alt="image-20230718185106666" style="zoom:67%;" />

> 起始就是快速幂的思想
>
> - 有一个base, 每次右移的时候都得翻倍
> - \> 0 乘, <0 除
> - 快速幂来就OK了

```go
func myPow(x float64, n int) float64 {     
	if n == 0 {
		return 1
	}
	base := x
	tmpN := n
	var res float64 = 1.0
	if n < 0 {
		tmpN = -n
	}
	for tmpN != 0 {
		if tmpN&1 != 0 {
			if n > 0 {
				res *= base
			} else {
				res /= base
			}

		}
		tmpN >>= 1
		base *= base
	}
	return res
}
```



## [48. 旋转图像](https://leetcode.cn/problems/rotate-image/)

<img src="./%E6%95%B0%E5%AD%A6-%E5%8D%9A%E5%BC%88-%E8%AE%A1%E7%BB%84-OS%E7%AD%89.assets/image-20230718211511814.png" alt="image-20230718211511814" style="zoom:67%;" />

> - 这种题找翻转肯定比找规律更简单(矩阵翻转解决很多问题)
> - 而且Golang中的多变量赋值肯定使得这种操作更简单
> - 先将其通过水平轴翻转, 根据主对角线翻转(66666666666666)

```go
func rotate(matrix [][]int) {
    n := len(matrix)
    // 水平翻转
    for i := 0; i < n/2; i++ {
        matrix[i], matrix[n-1-i] = matrix[n-1-i], matrix[i]
    }
    // 主对角线翻转
    for i := 0; i < n; i++ {
        for j := 0; j < i; j++ {
            matrix[i][j], matrix[j][i] = matrix[j][i], matrix[i][j]
        }
    }
}
```

## [43. 字符串相乘](https://leetcode.cn/problems/multiply-strings/)

<img src="./%E6%95%B0%E5%AD%A6-%E5%8D%9A%E5%BC%88-%E8%AE%A1%E7%BB%84-OS%E7%AD%89.assets/image-20230718212444623.png" alt="image-20230718212444623" style="zoom:67%;" />

> 真正的解法远比自己想象的easy
>
> 而且不需要二维数组, 一维数组循环相加就ok了

<img src="./%E6%95%B0%E5%AD%A6-%E5%8D%9A%E5%BC%88-%E8%AE%A1%E7%BB%84-OS%E7%AD%89.assets/image-20230718212524720.png" alt="image-20230718212524720" style="zoom:67%;" />

```go
func multiply(num1 string, num2 string) string {
	if num1 == "0" || num2 == "0" {
		return "0"
	}

	l1, l2 := len(num1), len(num2)
	cal := make([]int, l1+l2+1, l1+l2+1)

	for i := 0; i < l1; i++ {
		for j := 0; j < l2; j++ {
			cal[i+j+1] += int(num1[i]-'0') * int(num2[j]-'0')
		}
	}
	jw := 0
	for i := l1 + l2; i >= 0; i-- {
		cal[i] = cal[i] + jw
		jw = cal[i] / 10
		cal[i] = cal[i] % 10
	}
	start := 0
	for i := 0; i < l1+l2+1; i++ {
		if cal[i] > 0 {
			start = i
			break
		}
	}
	var res bytes.Buffer
	for i := start; i < l1+l2; i++ {
		res.WriteByte(byte(cal[i] + '0'))
	}

	return res.String()
}
```

## [69. x 的平方根 ](https://leetcode.cn/problems/sqrtx/)

<img src="./%E6%95%B0%E5%AD%A6-%E5%8D%9A%E5%BC%88-%E8%AE%A1%E7%BB%84-OS%E7%AD%89.assets/image-20230720112241243.png" alt="image-20230720112241243" style="zoom:67%;" />

> 二分还是得向之前的模板靠拢, 要不然边界条件容易错
>
> 一直是 l=mid+1, r=mid

```go
func mySqrt(x int) int {
	if x <= 1 {
		return x
	}
	l, r := 0, x
	for l < r {
		mid := l + (r-l)/2
		if x/mid >= mid {
			l = mid + 1
		} else {
			r = mid
		}
	}
	return l - 1
}
```

> 牛顿迭代法:
>
> <img src="./%E6%95%B0%E5%AD%A6-%E5%8D%9A%E5%BC%88-%E8%AE%A1%E7%BB%84-OS%E7%AD%89.assets/image-20230720115412086.png" alt="image-20230720115412086" style="zoom:50%;" />
>
> <img src="./%E6%95%B0%E5%AD%A6-%E5%8D%9A%E5%BC%88-%E8%AE%A1%E7%BB%84-OS%E7%AD%89.assets/image-20230720115443469.png" alt="image-20230720115443469" style="zoom:50%;" />

<img src="./%E6%95%B0%E5%AD%A6-%E5%8D%9A%E5%BC%88-%E8%AE%A1%E7%BB%84-OS%E7%AD%89.assets/04a77f809f58ddce5c0e8cabc225dc4.jpg" alt="04a77f809f58ddce5c0e8cabc225dc4" style="zoom: 33%;" />

```go
func mySqrt(x int) int {
	if x <= 1 {
		return x
	}
	y := x
	for x/y < y {
		y = (y + x/y) / 2
	}
	return y
}
```

## [118. 杨辉三角](https://leetcode.cn/problems/pascals-triangle/)

<img src="./%E6%95%B0%E5%AD%A6-%E5%8D%9A%E5%BC%88-%E8%AE%A1%E7%BB%84-OS%E7%AD%89.assets/image-20230724092404898.png" alt="image-20230724092404898" style="zoom:67%;" />

> 加法来写的

```go
func generate(numRows int) [][]int {
	ans := make([][]int, numRows)
	for i := range ans {
		ans[i] = make([]int, i+1) // 这样直接设定了长度哈
		ans[i][0] = 1
		ans[i][i] = 1
		for j := 1; j < i; j++ {
			ans[i][j] = ans[i-1][j] + ans[i-1][j-1]
		}
	}
	return ans
}
```

## [119. 杨辉三角 II](https://leetcode.cn/problems/pascals-triangle-ii/)

<img src="./%E6%95%B0%E5%AD%A6-%E5%8D%9A%E5%BC%88-%E8%AE%A1%E7%BB%84-OS%E7%AD%89.assets/image-20230724093513175.png" alt="image-20230724093513175" style="zoom:67%;" />

> 两个数组一直叫唤呗

```go
func getRow(rowIndex int) []int {
	res := make([]int, 34, 34)
	tmp := make([]int, 34, 34)
	res[0] = 1
	for i := 0; i < rowIndex; i++ {
		tmp[0] = 1
		for j := 1; j <= rowIndex; j++ {
			tmp[j] = res[j-1] + res[j]
		}
		tmp, res = res, tmp
	}
	return res[:rowIndex+1]
}
```

## [149. 直线上最多的点数](https://leetcode.cn/problems/max-points-on-a-line/)

<img src="./%E6%95%B0%E5%AD%A6-%E5%8D%9A%E5%BC%88-%E8%AE%A1%E7%BB%84-OS%E7%AD%89.assets/image-20230725230455109.png" alt="image-20230725230455109" style="zoom:67%;" />

> 枚举直线 + 枚举统计
>
> 因此一个朴素的做法是先枚举两条点（确定一条线），然后检查其余点是否落在该线中。
>
> 为了避免除法精度问题，当我们枚举两个点 i 和 j 时，不直接计算其对应直线的 斜率和 截距，而是通过判断 i 和 j 与第三个点 k 形成的两条直线斜率是否相等（斜率相等的两条直线要么平行，要么重合，平行需要 4 个点来唯一确定，我们只有 3 个点，所以可以直接判定两直线重合）。
>
> 这就是3个for循环啊
>
> - 时间复杂度：O(n3)
> - 空间复杂度：O(1)

```go
func maxPoints(points [][]int) int {
	n := len(points)
	ans := 1
	for i := 0; i < n; i++ {
		x := points[i]
		for j := i + 1; j < n; j++ {
			y := points[j]
			cnt := 2
			for k := j + 1; k < n; k++ {
				z := points[k]
                // 乘法代替除法 Good
				s1 := (x[0] - y[0]) * (y[1] - z[1])
				s2 := (x[1] - y[1]) * (y[0] - z[0])
				if s1 == s2 {
					cnt++
				}
			}
			ans = Max(ans, cnt)
		}
	}
	return ans
}
```

> 具体的，我们可以先枚举所有可能出现的 直线斜率（根据两点确定一条直线，即枚举所有的「点对」），使用「哈希表」统计所有 斜率 对应的点的数量，在所有值中取个 max 即是答案。
>
> 统计斜率里面有多少个点

```go
func maxPoints(points [][]int) int {
	// 斜率计算
	lineSlope := func(a, b []int) float64 {
		return float64(a[1]-b[1]) / float64(a[0]-b[0]) //相当于(ay-by) / (ax-bx)
	}

	//以i点为枢纽，找到与i点组成的直线的斜率相同的点
	res := 0 //返回值
	for i := 0; i < len(points); i++ {
		//建立一个key为两点斜率，可能为浮点数，所以才要float64，value为相同斜率点的个数
		hash := make(map[float64]int)
		for j := 0; j < len(points); j++ {
			if i != j { //不能是同一个点组成直线
				hash[lineSlope(points[i], points[j])]++
			}
		}
		// 注意这是在第一层for的里面
		for _, v := range hash {
			res = max(res, v)
		}
	}
	return res + 1 //因为hash表没有统计点i它自身，所以需要加一
}
```



## [168. Excel表列名称](https://leetcode.cn/problems/excel-sheet-column-title/)

<img src="./%E6%95%B0%E5%AD%A6-%E5%8D%9A%E5%BC%88-%E8%AE%A1%E7%BB%84-OS%E7%AD%89.assets/image-20230725235839831.png" alt="image-20230725235839831" style="zoom:67%;" />

> 切勿眼高手低
>
> 我们调整对应关系，让0对应A，25对应Z
>
> // 以z为特例, 就知道怎么操作了

```go
func convertToTitle(columnNumber int) string {
	res := ""
	for columnNumber != 0 {
		res = string(rune('A'+(columnNumber-1)%26)) + res
		columnNumber = (columnNumber - 1) / 26
	}
	return res
}
```









## [171. Excel 表列序号](https://leetcode.cn/problems/excel-sheet-column-number/)

<img src="./%E6%95%B0%E5%AD%A6-%E5%8D%9A%E5%BC%88-%E8%AE%A1%E7%BB%84-OS%E7%AD%89.assets/image-20230725235445420.png" alt="image-20230725235445420" style="zoom:67%;" />

> 切勿眼高手低

```go
func titleToNumber(columnTitle string) int {
	res := 0
	for idx := 0; idx < len(columnTitle); idx++ {
		res = res*26 + int(columnTitle[idx]-'A') + 1
	}
	return res
}
```

## [172. 阶乘后的零](https://leetcode.cn/problems/factorial-trailing-zeroes/)

<img src="./%E6%95%B0%E5%AD%A6-%E5%8D%9A%E5%BC%88-%E8%AE%A1%E7%BB%84-OS%E7%AD%89.assets/image-20230726104044027.png" alt="image-20230726104044027" style="zoom:67%;" />

> **进阶：**你可以设计并实现对数时间复杂度的算法来解决此问题吗？

```go
func trailingZeroes(n int) int {
	res := 0
	jc := 1
	for i := 2; i <= n; i++ {
		jc *= i
		for jc%10 == 0 {
			res += 1
			jc /= 10
		}
        // 不能太大, 不能太小, 防止溢出, 又不能失去精度
		if jc > 100000 {
			jc %= 100000
		}
	}
	return res
}
```

> <img src="./%E6%95%B0%E5%AD%A6-%E5%8D%9A%E5%BC%88-%E8%AE%A1%E7%BB%84-OS%E7%AD%89.assets/image-20230726104656730.png" alt="image-20230726104656730" style="zoom: 67%;" />

```go
func trailingZeroes(n int) (ans int) {
	for i := 5; i <= n; i += 5 {
		for x := i; x%5 == 0; x /= 5 {
			ans++
		}
	}
	return
}  // 比我之前的方法好很多啊
```

> <img src="./%E6%95%B0%E5%AD%A6-%E5%8D%9A%E5%BC%88-%E8%AE%A1%E7%BB%84-OS%E7%AD%89.assets/image-20230726104923913.png" alt="image-20230726104923913" style="zoom:67%;" />

```go
func trailingZeroes(n int) (ans int) {
	for n != 0 {
		n /= 5
		ans += n
	}
	return
}
```

```c++
class Solution {
public:
    int trailingZeroes(int n) {
        return n/5 + n/25 + n/125 + n/625 + n/3125 + n/15625;
    }
};
```



## [166. 分数到小数](https://leetcode.cn/problems/fraction-to-recurring-decimal/)

<img src="./%E6%95%B0%E5%AD%A6-%E5%8D%9A%E5%BC%88-%E8%AE%A1%E7%BB%84-OS%E7%AD%89.assets/image-20230726172518578.png" alt="image-20230726172518578" style="zoom:67%;" />

<img src="./%E6%95%B0%E5%AD%A6-%E5%8D%9A%E5%BC%88-%E8%AE%A1%E7%BB%84-OS%E7%AD%89.assets/9ed591db027164bae89e828b20960b6b_1633223480-OLGSxy-file_1633223479713.jpg" alt="9ed591db027164bae89e828b20960b6b_1633223480-OLGSxy-file_1633223479713" style="zoom:67%;" />

> 余数重复妈的就是循环了, fkfkfkfkfk
>
> 失控复杂度 都是 O(l)
>
> 真的是66666呀

```go
func fractionToDecimal(numerator int, denominator int) string {
	// 直接返回
	if numerator%denominator == 0 {
		return strconv.Itoa(numerator / denominator)
	}

	// 返回的值
	res := ""

	// 判断负数
	if numerator*denominator < 0 {
		res += "-"
	}
	numerator, denominator = abs(numerator), abs(denominator)

	// 直接计算整数部分
	res += strconv.Itoa(numerator/denominator) + "."
	numerator %= denominator

	// 记录余数的部分, 记录的是余数所在的位置
	m := map[int]int{}

	// 计算余数
	for numerator != 0 {
		m[numerator] = len(res)
		numerator *= 10
		res += strconv.Itoa(numerator / denominator)
		numerator %= denominator
		if m[numerator] != 0 {
			return fmt.Sprintf("%s(%s)", res[:m[numerator]], res[m[numerator]:len(res)])
		}
	}
	return res
}
```

## [202. 快乐数](https://leetcode.cn/problems/happy-number/)

<img src="./%E6%95%B0%E5%AD%A6-%E5%8D%9A%E5%BC%88-%E8%AE%A1%E7%BB%84-OS%E7%AD%89.assets/image-20230728113131818.png" alt="image-20230728113131818" style="zoom:67%;" />

```go
func isHappy(n int) bool {
	m := map[int]bool{}
	for {
		s := 0
		for n != 0 {
			t := n % 10
			m[t] = true
			s += (t) * (t)
			n /= 10
		}
		if s == 1 {
			return true
		}else if m[s]{
			return false
		}else{
			m[s] = true
		}
		n = s
	}
}
```

> 快慢指针法
>
> 这个过程可以看做是一个隐式的链表, 判断这个链表是否有环

```go
func isHappy(n int) bool {
	step := func(n int) int {
		sum := 0
		for ; n > 0; n /= 10 {
			sum += (n % 10) * (n % 10)
		}
		return sum
	}

	slow, fast := n, step(n)
	for fast != 1 && slow != fast {
		slow = step(slow)
		fast = step(step(fast))
	}
	return fast == 1
}
```



## [223. 矩形面积](https://leetcode.cn/problems/rectangle-area/)

<img src="./%E6%95%B0%E5%AD%A6-%E5%8D%9A%E5%BC%88-%E8%AE%A1%E7%BB%84-OS%E7%AD%89.assets/image-20230804120525460.png" alt="image-20230804120525460" style="zoom:67%;" />

```go
func computeArea(ax1, ay1, ax2, ay2, bx1, by1, bx2, by2 int) int {
	area1 := (ax2 - ax1) * (ay2 - ay1)
	area2 := (bx2 - bx1) * (by2 - by1)
	overlapWidth := min(ax2, bx2) - max(ax1, bx1)
	overlapHeight := min(ay2, by2) - max(ay1, by1)

	// 一定要和0进行比较啊
	overlapArea := max(overlapWidth, 0) * max(overlapHeight, 0)
	return area1 + area2 - overlapArea
}
```



## [1201. 丑数 III](https://leetcode.cn/problems/ugly-number-iii/)

<img src="./%E6%95%B0%E5%AD%A6-%E5%8D%9A%E5%BC%88-%E8%AE%A1%E7%BB%84-OS%E7%AD%89.assets/image-20230830161510442.png" alt="image-20230830161510442" style="zoom:67%;" />

> 二分查找, 看三个因子各包含了多少个
>
> 直接使用熔池订立

```c++
class Solution {
using ll = long long;
ll gcd(ll a, ll b){
	return b==0 ? a : gcd(b, a%b);	
}

ll lcm(ll a, ll b){
	return a*b/gcd(a,b);
}

public:
    int nthUglyNumber(int n, int a, int b, int c) {
        ll ab=lcm(a, b), ac=lcm(a,c), bc=lcm(b,c);
        ll abc = lcm(ab, c);
        ll l = min({a,b,c})-1, r=2e9+1;
        while (l  < r){
        	ll mid = (l+r)/2;
        	ll cnt = mid/a + mid/b + mid/c -mid/ab -mid/ac - mid/bc + mid/abc;
        	if(cnt<n) l=mid+1;
        	else r=mid;
	    }
	    return static_cast<int>(r);
	}
};
```

## [775. 全局倒置与局部倒置](https://leetcode.cn/problems/global-and-local-inversions/)

<img src="./%E6%95%B0%E5%AD%A6-%E5%8D%9A%E5%BC%88-%E8%AE%A1%E7%BB%84-OS%E7%AD%89.assets/image-20230830163255586.png" alt="image-20230830163255586" style="zoom:67%;" />

> 0-n-1的一个排序
>
> - 有一个局部, 就一定有一个全局
> - 看隔开的有没有逆序

```c++
class Solution {
   public:
    bool isIdealPermutation(vector<int>& nums) {
        int n = nums.size(), minRecord = nums[n - 1];
        for (int i = n - 3; i >= 0; i--) {
            if (nums[i] > minRecord) return false;
            minRecord = min(minRecord, nums[i + 1]);
        }
        return true;
    }
```



## **P1143 进制转换**

<img src="./%E6%95%B0%E5%AD%A6-%E5%8D%9A%E5%BC%88-%E8%AE%A1%E7%BB%84-OS%E7%AD%89.assets/image-20230904162359005.png" alt="image-20230904162359005" style="zoom:67%;" />

```c++
int main() {
    ll n, m, ten{};
    string s, table{"0123456789ABCDEF"};
    cin >> n >> s >> m;

    for (auto c : s) {
        ten = ten * n + (c >= 'A' ? c - 'A' + 10 : c - '0');
    }
    vector<char> res{};
    while(ten){
    	res.push_back(table[ten%m]);
		ten /= m;	
    }
    reverse(res.begin(), res.end());
    for(auto i: res)
    	cout << i;
    	
    if(res.size()==0) cout<<'0'; 
    	
    return 0;
}
```

## P1017 [NOIP2000 提高组] 进制转换

<img src="./%E6%95%B0%E5%AD%A6-%E5%8D%9A%E5%BC%88-%E8%AE%A1%E7%BB%84-OS%E7%AD%89.assets/image-20230904162524180.png" alt="image-20230904162524180" style="zoom:67%;" />

```c++
int main() {
    int n, R, n_tmp;
    vector<char> res;
    string idx = "0123456789ABCDEFGHIJKLMNOPQ";
    cin >> n >> R;
    n_tmp = n;
    while (n) {
        int j = n % R;
        n /= R;
        if (j < 0) {
            j -= R;
            n++;
        }
        res.push_back(idx[j]);
    }
    cout << n_tmp << "=";
    for (auto i = res.rbegin(); i < res.rend(); i++) cout << *i;
    cout << "(base" << R << ")";
    return 0;
}
```

## P2822 [NOIP2016 提高组] 组合数问题

<img src="./%E6%95%B0%E5%AD%A6-%E5%8D%9A%E5%BC%88-%E8%AE%A1%E7%BB%84-OS%E7%AD%89.assets/image-20230904162713072.png" alt="image-20230904162713072" style="zoom:67%;" />

```c++
// 使用的是A的公式
ll C(int n, int m) {
    ll anm{1}, amm{1};
    for (int i = n - m + 1; i <= n; i++) anm *= i;
    for (int i = 2; i <= m; i++) amm *= i;
    return anm / amm;
}

int main() {
    int t, k;
    cin >> t >> k;
    for (int fff = 0; fff < t; fff++) {
        int n, m, res{};
        cin >> n >> m;
        for (int i = 0; i <= n; i++) {
            for (int j = 0; j <= min(i, m); j++) {
                if ((C(i, j) % k) == 0) {
                    res += 1;
                }
            }
        }
        cout << res << endl;
    }
    return 0;
}
```

````c++
// 递归公式＋记忆
ll C(int n, int m) {
    string s = to_string(n) + " " + to_string(m);
    if (mp.count(s)) return mp[s];
    if (n == 0 && m == 0) return 0;
    if (n == 0 || m == 0) return 1;
    if (n == m) return 1;
    ll res = C(n - 1, m - 1) + C(n - 1, m);
    if (!mp.count(s)) mp[s] = res;
    return res;
}

int main() {
    cin.tie(0);
    ios::sync_with_stdio(0);
    int t, k;
    cin >> t >> k;
    for (int fff = 0; fff < t; fff++) {
        int n, m, res{};
        cin >> n >> m;
        for (int i = 0; i <= n; i++) {
            for (int j = 0; j <= min(i, m); j++) {
                if ((C(i, j) % k) == 0 && (C(i, j) >= k)) {
                    res += 1;
                }
            }
        }
        cout << res << endl;
    }
    return 0;
}
````

```c++
// 数组模拟递推
using ll = long long;
ll YH[2001][2001];

int main() {
    cin.tie(0);
    ios::sync_with_stdio(0);
    int t, k;
    cin >> t >> k;

    YH[0][0] = YH[1][0] = YH[1][1] = 1;
    for (int i = 2; i <= 2000; i++) {
        YH[i][0] = 1;
        for (int j = 1; j <= i; j++)
            YH[i][j] = (YH[i - 1][j] % k + YH[i - 1][j - 1] % k) % k;
    }

    for (int fff = 0; fff < t; fff++) {
        int n, m, res{};
        cin >> n >> m;
        for (int i = 0; i <= n; i++) {
            for (int j = 0; j <= min(i, m); j++) {
                if (YH[i][j] % k == 0) {
                    res++;
                }
            }
        }
        cout << res << endl;
    }
    return 0;
}
```

```c++
// 使用前缀和的方法
using ll = long long;
ll YH[2001][2001];
int ans[2001][2001];

int main() {
    cin.tie(0);
    ios::sync_with_stdio(0);
    int t, k;
    cin >> t >> k;

    YH[0][0] = YH[1][0] = YH[1][1] = 1;
    for (int i = 2; i <= 2000; i++) {
        YH[i][0] = 1;
        for (int j = 1; j <= i; j++) {
            ans[i][j] = ans[i][j - 1] + ans[i - 1][j] - ans[i - 1][j - 1];
            YH[i][j] = (YH[i - 1][j] % k + YH[i - 1][j - 1] % k) % k;
            if (YH[i][j] == 0) ans[i][j]++;
        }
        ans[i][i + 1] = ans[i][i];
    }

    for (int fff = 0; fff < t; fff++) {
        int n, m, res{};
        cin >> n >> m;
        res = m > n ? ans[n][n] : ans[n][m];
        cout << res << endl;
    }
    return 0;
}
```





## [829/83. 连续整数求和](https://leetcode.cn/problems/consecutive-numbers-sum/)

<img src="./%E6%95%B0%E5%AD%A6-%E5%8D%9A%E5%BC%88-%E8%AE%A1%E7%BB%84-OS%E7%AD%89.assets/image-20231127222012332.png" alt="image-20231127222012332" style="zoom:67%;" />

> - 一看就是求这个数字的因子, 因为连续整数至与中位数有关
>   - 中位数要么是.5, 要么是整数
>   - 1个数时，必然有一个数可构成N
>   - 2个数若要构成N，第2个数与第1个数差为1，N减掉这个1能整除2则能由商与商+1构成N'
>   - 3个数若要构成N，第2个数与第1个数差为1，第3个数与第1个数的差为2，N减掉1再减掉2能整除3则能由商、商+1与商+2构成N

```go
// 这个解法是真的喵喵啊
func consecutiveNumbersSum(n int) int {
	res := 0
	start := 1
	for n > 0 {
		if (n-start)%start == 0 {
			res++
		}
		n -= start
		start++
	}
	return res
}
```



# LCM & GCD

## ~~[858/90. 镜面反射](https://leetcode.cn/problems/mirror-reflection/)~~

<img src="./%E6%95%B0%E5%AD%A6-%E5%8D%9A%E5%BC%88-%E8%AE%A1%E7%BB%84-OS%E7%AD%89.assets/image-20231213102851097.png" alt="image-20231213102851097" style="zoom:67%;" />



> 就是看纵方向上走了P的整数倍的时候走了多少个q
>
> - 偶数个 => 到2
> - 奇数个
>   - 路径走了奇数个p => 到1
>   - 偶数个 => 到0

```GO
func mirrorReflection(p int, q int) int {
	lcm := p / gcd(p, q) * q
	if (lcm/q)%2 == 0 {
		return 2
	} else if (lcm/p)%2 == 0 {
		return 0
	}
	return 1
}

func gcd(i, j int) int {
	if j == 0 {
		return i
	}
	return gcd(j, i%j)
}
```











# 质数

## [204. 计数质数](https://leetcode.cn/problems/count-primes/)

<img src="./%E6%95%B0%E5%AD%A6-%E5%8D%9A%E5%BC%88-%E8%AE%A1%E7%BB%84-OS%E7%AD%89.assets/image-20230728115744453.png" alt="image-20230728115744453" style="zoom:67%;" />

> 埃氏筛
>
> 欧拉筛

```go
func countPrimes(n int) (cnt int) {
	isPrime := make([]bool, n)
	for i := range isPrime {
		isPrime[i] = true
	}
	for i := 2; i < n; i++ {
		if isPrime[i] {
			cnt++
			for j := 2 * i; j < n; j += i {
				isPrime[j] = false
			}
		}
	}
	return cnt
}
```

```go
func countPrimes(n int) int {
	primes := []int{}
	isPrime := make([]bool, n)
	for i := range isPrime {
		isPrime[i] = true
	}
	for i := 2; i < n; i++ {
		if isPrime[i] {
			primes = append(primes, i)
		}
		for _, p := range primes {
			if i*p >= n {
				break
			}
			isPrime[i*p] = false
			if i%p == 0 {
				break
			}
		}
	}
	return len(primes)
}
```



## [786. 第 K 个最小的素数分数](https://leetcode.cn/problems/k-th-smallest-prime-fraction/)

<img src="./%E6%95%B0%E5%AD%A6-%E5%8D%9A%E5%BC%88-%E8%AE%A1%E7%BB%84-OS%E7%AD%89.assets/image-20230830152630557.png" alt="image-20230830152630557" style="zoom:67%;" />

> 1. 直接暴力＋自定义排序 就行了 
> 2. 优先队列
>    1. <img src="./%E6%95%B0%E5%AD%A6-%E5%8D%9A%E5%BC%88-%E8%AE%A1%E7%BB%84-OS%E7%AD%89.assets/image-20230830153207245.png" alt="image-20230830153207245" style="zoom:50%;" />
>    2. 这个真的很妙啊, 相当于多路归并, 出去一个就把这条路的下一个放入堆.

```go
func kthSmallestPrimeFraction(arr []int, k int) []int {
	n := len(arr)
	h := make(hp, n-1)
	for j := 1; j < n; j++ {
		h[j-1] = frac{arr[0], arr[j], 0, j}
	}
	heap.Init(&h)
	for loop := k - 1; loop > 0; loop-- {
		f := heap.Pop(&h).(frac)
		if f.i+1 < f.j {
			heap.Push(&h, frac{arr[f.i+1], f.y, f.i + 1, f.j})
		}
	}
	return []int{h[0].x, h[0].y}
}

type frac struct{ x, y, i, j int }
type hp []frac
func (h hp) Len() int            { return len(h) }
func (h hp) Less(i, j int) bool  { return h[i].x*h[j].y < h[i].y*h[j].x }
func (h hp) Swap(i, j int)       { h[i], h[j] = h[j], h[i] }
func (h *hp) Push(v interface{}) { *h = append(*h, v.(frac)) }
func (h *hp) Pop() interface{}   { a := *h; v := a[len(a)-1]; *h = a[:len(a)-1]; return v }
```



## [1250. 检查「好数组」](https://leetcode.cn/problems/check-if-it-is-a-good-array/)

<img src="./%E6%95%B0%E5%AD%A6-%E5%8D%9A%E5%BC%88-%E8%AE%A1%E7%BB%84-OS%E7%AD%89.assets/image-20230830162824044.png" alt="image-20230830162824044" style="zoom:67%;" />

> 裴蜀定理

```c++
class Solution {
using ll = long long;
ll gcd(ll a, ll b){
	return b==0 ? a : gcd(b, a%b); 
}

public:
    bool isGoodArray(vector<int>& nums) {
    	int res {nums[0]};
		for(int i=1; i<nums.size(); i++){
			res = gcd(res, nums[i]);
			if(res==1) return true;
		}
		return res==1;
    }
};
```





# 位运算

## [136. 只出现一次的数字](https://leetcode.cn/problems/single-number/)

<img src="./%E6%95%B0%E5%AD%A6-%E5%8D%9A%E5%BC%88-%E8%AE%A1%E7%BB%84-OS%E7%AD%89.assets/image-20230724152854844.png" alt="image-20230724152854844" style="zoom:67%;" />

> 原地算法, 可以考虑位运算

```go
func singleNumber(nums []int) int {
	res := nums[0]
	for i := 1; i < len(nums); i++ {
		res ^= nums[i]
	}
	return res
}
```

## [137. 只出现一次的数字 II](https://leetcode.cn/problems/single-number-ii/)

<img src="./%E6%95%B0%E5%AD%A6-%E5%8D%9A%E5%BC%88-%E8%AE%A1%E7%BB%84-OS%E7%AD%89.assets/image-20230724153254792.png" alt="image-20230724153254792" style="zoom:67%;" />

> 这题之前没做过
>
> 怎么使用位运算呢??????????

> 把第一个数转成三进制，然后对后面的数，依次转成三进制，并按位叠加到第一个数上，叠加时不做进位操作（对每一个三进制位，有0+x=x，1+1=2，2+1=0，2+2=1这样的运算）
>
> 最后得到的结果，转回正常的十进制数就是答案。
>
> 之前的题目相当于转换为二进制的和是0 Good
>
> 
>
> 这样写边界条件太复杂了!!!!!!!!!!!

```go
func singleNumber(nums []int) int {
	threeAdd := func(s, t string) string {
		// 对于负数的特殊处理
        // 长度延伸至32位
		if s[0] == '-' {
			s = "1" + strings.Repeat("0", 32-len(s)) + s[1:]
		}
		if t[0] == '-' {
			t = "1" + strings.Repeat("0", 32-len(t)) + t[1:]
		}
        // 交换好对齐
		if len(s) > len(t) {
			s, t = t, s
		}
		res := ""
		// 对齐
		s = strings.Repeat("0", len(t)-len(s)) + s
        // 相加
		for i := 0; i < len(s); i++ {
			res += strconv.Itoa(int(s[i]-'0'+t[i]-'0') % 3)
		}
		return res
	}
	
	res := strconv.FormatInt(int64(nums[0]), 3)
	for i := 1; i < len(nums); i++ {
		res = threeAdd(res, strconv.FormatInt(int64(nums[i]), 3))
		fmt.Println(res)
	}

    // 判断是不是负数
	flag := 1
	if len(res) == 32 && res[0] == '1' {
		flag = -1
		res = res[1:]
	}
	
    // 直接转换
	ret, _ := strconv.ParseInt(res, 3, 0)
	return int(ret) * flag
}
```

> 一个二进制位只能表示0或者1。也就是天生可以记录一个数出现了一次还是两次。
>
> - x ^ 0 = x;
> - x ^ x = 0;
>
> 要记录出现3次，需要两个二进制位。那么上面单独的`x`就不行了。我们需要两个变量，每个变量取一位：
>
> - ab ^ 00 = ab;
> - ab ^ ab = 00;
>
> 这里，`a`、`b`都是32位的变量。我们使用`a`的第`k`位与`b`的第`k`位组合起来的两位二进制，表示当前位出现了几次。也就是，一个`8`位的二进制`x`就变成了`16`位来表示。
>
> 它是一个逻辑电路，`a`、`b`变量中，相同位置上，分别取出一位，负责完成`00->01->10->00`，也就是开头的那句话，当数字出现3次时置零。

<img src="./%E6%95%B0%E5%AD%A6-%E5%8D%9A%E5%BC%88-%E8%AE%A1%E7%BB%84-OS%E7%AD%89.assets/image-20230724161656317.png" alt="image-20230724161656317" style="zoom:67%;" />

<img src="./%E6%95%B0%E5%AD%A6-%E5%8D%9A%E5%BC%88-%E8%AE%A1%E7%BB%84-OS%E7%AD%89.assets/image-20230724161715615.png" alt="image-20230724161715615" style="zoom:67%;" />

```go
// 先计算b, 那么a的更新规律就简单啦
func singleNumber(nums []int) int {
	a, b := 0, 0
	for _, num := range nums {
		b = (b ^ num) &^ a
		a = (a ^ num) &^ b
	}
	return b
}
```

> <img src="./%E6%95%B0%E5%AD%A6-%E5%8D%9A%E5%BC%88-%E8%AE%A1%E7%BB%84-OS%E7%AD%89.assets/image-20230724162010223.png" alt="image-20230724162010223" style="zoom:67%;" />
>
> 进行跟多轮, 每次确定一个二进制位.....哈哈哈

```go
func singleNumber(nums []int) int {
    ans := int32(0)
    for i := 0; i < 32; i++ {
        total := int32(0)
        for _, num := range nums {
            total += int32(num) >> i & 1
        }
        if total%3 > 0 {
            ans |= 1 << i
        }
    }
    return int(ans)
}
```



## [191. 位1的个数](https://leetcode.cn/problems/number-of-1-bits/)

<img src="./%E6%95%B0%E5%AD%A6-%E5%8D%9A%E5%BC%88-%E8%AE%A1%E7%BB%84-OS%E7%AD%89.assets/image-20230726113248882.png" alt="image-20230726113248882" style="zoom:67%;" />

> 循环遍历二进制位

```go
func hammingWeight(num uint32) int {
	res := 0
	for i:=0; i<32; i++{
		if num & (1<<i) != 0{
			res ++
		}
	}
	return res
}
```

> $n\&n-1$ 把二进制最低位变为0

```go
func hammingWeight(num uint32) int {
	res := 0
	for num != 0 {
		res += 1
		num &= num - 1
	}
	return res
}
```

> hamming

```go
func hammingWeight(num uint32) int {
	// HD, Fnumgure 5-2
	num = num - ((num >> 1) & 0x55555555)
	num = (num & 0x33333333) + ((num >> 2) & 0x33333333)
	num = (num + (num >> 4)) & 0x0f0f0f0f
	num = num + (num >> 8)
	num = num + (num >> 16)
	return int(num & 0x3f)
}
```

<img src="./%E6%95%B0%E5%AD%A6-%E5%8D%9A%E5%BC%88-%E8%AE%A1%E7%BB%84-OS%E7%AD%89.assets/image-20230726114003789.png" alt="image-20230726114003789" style="zoom:67%;" />

<img src="./%E6%95%B0%E5%AD%A6-%E5%8D%9A%E5%BC%88-%E8%AE%A1%E7%BB%84-OS%E7%AD%89.assets/image-20230726114041243.png" alt="image-20230726114041243" style="zoom:67%;" />

第一步：每2位内部合并一个结果；
第二步：每4位内部合并一个结果；
第三步：每8位内部合并一个结果；
第四步：每16位内部合并一个结果；
最后：每32位内部合并一个结果，得到答案。

```go
func hammingWeight(n uint32) int {
	n = (n & 0x55555555) + ((n & 0xaaaaaaaa) >> 1)
	n = (n & 0x33333333) + ((n & 0xcccccccc) >> 2)
	n = (n & 0x0f0f0f0f) + ((n & 0xf0f0f0f0) >> 4)
	n = (n & 0x00ff00ff) + ((n & 0xff00ff00) >> 8)
	n = (n & 0x0000ffff) + ((n & 0xffff0000) >> 16)
	return int(n)
}
```

```go
func hammingWeight(n uint32) int {
	n = (n & 0x55555555) + ((n >> 1) & 0x55555555)
	n = (n & 0x33333333) + ((n >> 2) & 0x33333333)
	n = (n & 0x0f0f0f0f) + ((n >> 4) & 0x0f0f0f0f)
	n = (n & 0x00ff00ff) + ((n >> 8) & 0x00ff00ff)
	n = (n & 0x0000ffff) + ((n >> 16) & 0x0000ffff)
	return int(n)
}
```

```java
// 注意下面的推导没有考虑运算符的优先级，通过空格来区别优先级
n&(0x55555555 + 0xaaaaaaaa)          == n
n&0x55555555 + n&0xaaaaaaaa          == n
n&0x55555555 + 2[(n&0xaaaaaaaa)>>>1] == n
n&0x55555555 + (n&0xaaaaaaaa)>>>1 == n - (n&0xaaaaaaaa)>>>1
//因此：
n&0x55555555 + ((n>>>1)&0x55555555) == n - ((n>>>1)&0x55555555)
```

> 思考：为什么方法5中前两行不可以把公共的`0x55555555/0x33333333`提出来呢？
>
> 应该是，这时n还比较大，提出公共部分后，`n+(n>>>1)`可能溢出`int`的表示范围。
>
> `n + (n >>> 8)`按位与上`0x00ff00ff`的目的是让相应的高位变成0，
>
> 其实这时候高位变不变为0已经不重要了，因为我们只关心4个f的低位，
>
> 这4个f的位置加在一起就是最后的答案（答案只须取n最后的一个字节)。
>
> 最后只需要考虑低六位
>
> 牛蛙牛蛙

## [190. 颠倒二进制位](https://leetcode.cn/problems/reverse-bits/)

<img src="./%E6%95%B0%E5%AD%A6-%E5%8D%9A%E5%BC%88-%E8%AE%A1%E7%BB%84-OS%E7%AD%89.assets/image-20230726153928568.png" alt="image-20230726153928568" style="zoom:67%;" />

> 1. 正常的枚举,枚举过程中判断那是不是0, 提前终止
> 2.  位运算分治
> 3. 和190 一样的做法, 先交换2位, 4为, 8位....

```go
func reverseBits(n uint32) uint32 {
   const (
      m1 = 0x55555555
      m2 = 0x33333333
      m4 = 0x0f0f0f0f
      m8 = 0x00ff00ff
   )
   n = n>>1&m1 | n&m1<<1
   n = n>>2&m2 | n&m2<<2
   n = n>>4&m4 | n&m4<<4
   n = n>>8&m8 | n&m8<<8
   return n>>16 | n<<16
}
```

## [201. 数字范围按位与](https://leetcode.cn/problems/bitwise-and-of-numbers-range/)

<img src="./%E6%95%B0%E5%AD%A6-%E5%8D%9A%E5%BC%88-%E8%AE%A1%E7%BB%84-OS%E7%AD%89.assets/image-20230728114154959.png" alt="image-20230728114154959" style="zoom:67%;" />

> 相与, 只要有一个是0, 那么就是0 => 必须找到规律啊
>
> 对所有数字执行按位与运算的结果是所有对应二进制字符串的公共前缀再用零补上后面的剩余位。

> 将两个数字不断向右移动，直到数字相等，即数字被缩减为它们的公共前缀。然后，通过将公共前缀向左移动，将零添加到公共前缀的右边以获得最终结果。
>
> 这也太聪明了吧

```go
func rangeBitwiseAnd(left int, right int) int {
	cnt := 0
	for left != right {
		cnt += 1
		left, right = left>>1, right>>1
	}
	return left << cnt
}
```

> Brian Kernighan 算法
>
> 清除最右边的1
>
> 对大的进行清除, 直至小于等于小的
>
> 动手模拟几次就明白啦

```go
func rangeBitwiseAnd(left int, right int) int {
	for left < right {
		right &= right - 1
	}
	return right
}
```



## [231. 2 的幂](https://leetcode.cn/problems/power-of-two/)

<img src="./%E6%95%B0%E5%AD%A6-%E5%8D%9A%E5%BC%88-%E8%AE%A1%E7%BB%84-OS%E7%AD%89.assets/image-20230806001033930.png" alt="image-20230806001033930" style="zoom:67%;" />

```go
func isPowerOfTwo(n int) bool {
    return (n>0) && n & (n-1) == 0
}
```

## P6599 「EZEC-2」异或

<img src="./%E6%95%B0%E5%AD%A6-%E5%8D%9A%E5%BC%88-%E8%AE%A1%E7%BB%84-OS%E7%AD%89.assets/image-20230829220841178.png" alt="image-20230829220841178" style="zoom:67%;" />

> <img src="./%E6%95%B0%E5%AD%A6-%E5%8D%9A%E5%BC%88-%E8%AE%A1%E7%BB%84-OS%E7%AD%89.assets/image-20230829223833479.png" alt="image-20230829223833479" style="zoom: 50%;" />

```golang
var MOD int64 = 1000000007

func main() {
	in := bufio.NewReader(os.Stdin)
	out := bufio.NewWriter(os.Stdout)
	defer out.Flush()
	var t int
	fmt.Fscan(in, &t)

	for i := 0; i < t; i++ {
		var n, l int64
		fmt.Fscan(in, &n, &l)

		if n == 1 {
			fmt.Println(0)
		} else {
			var x = Wei(n)
			res := ((l / 2) % MOD) * ((l - l/2) % MOD) * (int64(math.Pow(float64(2), float64(x))-1) % MOD)
			fmt.Println(res % MOD)
		}
	}
}

func Wei(num int64) int {
	for i := 1; ; i++ {
		if num>>i == 0 {
			return i
		}
	}
}
```

## P8763 [蓝桥杯 2021 国 ABC] 异或变换

<img src="./%E6%95%B0%E5%AD%A6-%E5%8D%9A%E5%BC%88-%E8%AE%A1%E7%BB%84-OS%E7%AD%89.assets/image-20230829223938772.png" alt="image-20230829223938772" style="zoom:67%;" />

> 肯定是找规律, 找到循环节!~!!!!
>
> $2^x$一定存在一次循环

```py	
from sys import stdin, stdout

n, t = map(int, stdin.readline().split())
s = list(map(int, list(stdin.readline().strip())))
st = [0 for _ in range(n)]

cir = 1
while cir < n:
    cir *= 2
t %= cir

for _ in range(t):
    st[0] = s[0]
    for i in range(1, n):
        st[i] = s[i] ^ s[i-1]
    s = st.copy()

print(*s, sep='')
```



# 计组 OS

## [146. LRU 缓存](https://leetcode.cn/problems/lru-cache/)

<img src="./%E6%95%B0%E5%AD%A6-%E5%8D%9A%E5%BC%88-%E8%AE%A1%E7%BB%84-OS%E7%AD%89.assets/image-20230725000311308.png" alt="image-20230725000311308" style="zoom:67%;" />

> 哈希 + 双向链表 哈哈哈

```go
type Entry struct {
	prev, next *Entry
	k, v       int
}

type List struct {
	head, tail *Entry
	length     int
}

func (l *List) MoveToFront(e *Entry) {
	e.prev.next = e.next
	e.next.prev = e.prev
	e.next = l.head.next
	e.next.prev = e
	l.head.next = e
	e.prev = l.head
}

func (l *List) PushFront(e *Entry) {
	e.next = l.head.next
	e.next.prev = e
	l.head.next = e
	e.prev = l.head
	l.length += 1
}

func (l *List) PopBack() {
	l.tail.prev = l.tail.prev.prev
	l.tail.prev.next = l.tail
	l.length -= 1
}

type LRUCache struct {
	m   map[int]*Entry
	l   *List
	cap int
}

func Constructor(capacity int) LRUCache {
	head := &Entry{}
	tail := &Entry{}
	head.next = tail
	tail.prev = head
	return LRUCache{
		cap: capacity,
		m:   make(map[int]*Entry),
		l: &List{
			head:   head,
			tail:   tail,
			length: 0,
		}}
}

func (this *LRUCache) Get(key int) int {
	if e, ok := this.m[key]; ok {
		this.l.MoveToFront(e)
		return e.v
	} else {
		return -1
	}
}

func (this *LRUCache) Put(key int, value int) {
	if e, ok := this.m[key]; ok {
		e.v = value
		this.l.MoveToFront(e)
	} else if this.l.length < this.cap {
		t := &Entry{k: key, v: value}
		this.l.PushFront(t)
		this.m[key] = t
	} else {
		delete(this.m, this.l.tail.prev.k)
		this.l.PopBack()
		t := &Entry{k: key, v: value}
		this.l.PushFront(t)
		this.m[key] = t
	}
}
```

## [150. 逆波兰表达式求值](https://leetcode.cn/problems/evaluate-reverse-polish-notation/)

<img src="./%E6%95%B0%E5%AD%A6-%E5%8D%9A%E5%BC%88-%E8%AE%A1%E7%BB%84-OS%E7%AD%89.assets/image-20230725000953637.png" alt="image-20230725000953637" style="zoom: 67%;" />

> 就是这样来做的呗, 逆波兰表达式就是使用栈
>
> good !!!!!!!!!!!!

```go
func evalRPN(tokens []string) int {
	stack := []int{}

	for idx := 0; idx < len(tokens); idx++ {
		switch tokens[idx] {
		case "+":
			t := stack[len(stack)-2] + stack[len(stack)-1]
			stack = stack[:len(stack)-2]
			stack = append(stack, t)
		case "-":
			t := stack[len(stack)-2] - stack[len(stack)-1]
			stack = stack[:len(stack)-2]
			stack = append(stack, t)
		case "*":
			t := stack[len(stack)-2] * stack[len(stack)-1]
			stack = stack[:len(stack)-2]
			stack = append(stack, t)
		case "/":
			t := stack[len(stack)-2] / stack[len(stack)-1]
			stack = stack[:len(stack)-2]
			stack = append(stack, t)
		default:
			// 入栈
			t, _ := strconv.Atoi(tokens[idx])
			stack = append(stack, t)
		}
	}
	return stack[0]
}
```

## [224. 基本计算器](https://leetcode.cn/problems/basic-calculator/)

<img src="./%E6%95%B0%E5%AD%A6-%E5%8D%9A%E5%BC%88-%E8%AE%A1%E7%BB%84-OS%E7%AD%89.assets/image-20230804121923470.png" alt="image-20230804121923470" style="zoom:67%;" />

> 中缀转后缀 => 逆波兰表达式
>
> 中缀表达式转后缀表达式的方法：
> 1. 遇到操作数：直接输出（添加到后缀表达式中）
> 2. 栈为空时，遇到运算符，直接入栈
> 3. 遇到左括号：将其入栈
> 4. 遇到右括号：执行出栈操作，并将出栈的元素输出，直到弹出栈的是左括号，括号不输出。
> 5. 遇到其他运算符：加减乘除：弹出所有优先级大于或者等于该运算符的栈顶元素，然后将该运算符入栈
> 6. 最终将栈中的元素依次出栈，输出。
>
> 最难的是一元表达式的负号
>
> - `(-` 替换为 `(0-`，`(+` 替换为 `(0+`   woc

```go
func calculate(s string) int {
	// 预处理, 处理掉一元运算符
	s = strings.ReplaceAll(s, " ", "")
	s = strings.ReplaceAll(s, "(-", "(0-")
	s = strings.ReplaceAll(s, "(+", "(0+")
	if s[0] == '-' {
		s = "0" + s
	}

	// 运算符优先级, 默认是0, ' '是-1, 避免计数
	opt := map[byte]int{'(': 3, ')': 3, '*': 2, '/': 2, '+': 1, '-': 1}

	// 中缀转后缀
	var rplBuff []string         // 记录逆波兰表达式
	rplStack := arraystack.New() // 记录逆波兰
	for idx := 0; idx < len(s); idx++ {
		if opt[s[idx]] == 0 { // 操作数直接输出, 但是考虑多位
			t := int(s[idx] - '0')
			for idx+1 < len(s) && opt[s[idx+1]] == 0 {
				idx++
				t = t*10 + int(s[idx]-'0')
			}
			rplBuff = append(rplBuff, strconv.Itoa(t))
		} else if rplStack.Empty() || s[idx] == '(' { // 栈空,左括号 操作符入栈
			rplStack.Push(s[idx])
		} else if s[idx] == ')' { // 右括号一直出栈
			for {
				b, _ := rplStack.Pop()
				if b.(byte) == '(' {
					break
				}
				rplBuff = append(rplBuff, string(b.(byte)))
			}
		} else {
			for !rplStack.Empty() { //所有优先级大于或者等于该运算符的栈顶元素
				b, _ := rplStack.Peek()
				if opt[b.(byte)] >= opt[s[idx]] {
					rplBuff = append(rplBuff, string(b.(byte)))
					rplStack.Pop()
				} else {
					break
				}
			}
			rplStack.Push(s[idx])
		}
	}
	// 全部出站
	for !rplStack.Empty() {
		b, _ := rplStack.Pop()
		rplBuff = append(rplBuff, string(b.(byte)))
	}

	// 后缀计算
	for idx := 0; idx < len(rplBuff); idx++ {
		switch rplBuff[idx] {
		case "+":
			t1, _ := rplStack.Pop()
			t2, _ := rplStack.Pop()
			rplStack.Push(t2.(int) + t1.(int))
		case "-":
			t1, _ := rplStack.Pop()
			t2, _ := rplStack.Pop()
			rplStack.Push(t2.(int) - t1.(int))
		case "*":
			t1, _ := rplStack.Pop()
			t2, _ := rplStack.Pop()
			rplStack.Push(t2.(int) * t1.(int))
		case "/":
			t1, _ := rplStack.Pop()
			t2, _ := rplStack.Pop()
			rplStack.Push(t2.(int) / t1.(int))
		default:
			t, _ := strconv.Atoi(rplBuff[idx])
			rplStack.Push(t)
		}
	}
	res, _ := rplStack.Peek()
	return res.(int)
}
```



## [227. 基本计算器 II](https://leetcode.cn/problems/basic-calculator-ii/)

<img src="./%E6%95%B0%E5%AD%A6-%E5%8D%9A%E5%BC%88-%E8%AE%A1%E7%BB%84-OS%E7%AD%89.assets/image-20230804133501082.png" alt="image-20230804133501082" style="zoom:67%;" />

> 答案和上面的题目一样



## [241. 为运算表达式设计优先级](https://leetcode.cn/problems/different-ways-to-add-parentheses/)

<img src="./%E6%95%B0%E5%AD%A6-%E5%8D%9A%E5%BC%88-%E8%AE%A1%E7%BB%84-OS%E7%AD%89.assets/image-20230806225820281.png" alt="image-20230806225820281" style="zoom:67%;" />

> 相同的负号可能也有不同的优先级
>
> 分治 gogogo

```go
func diffWaysToCompute(expression string) []int {
	if unicode.IsDigit(rune(expression[0])) && len(expression) <= 2 {
		num, _ := strconv.Atoi(expression)
		return []int{num}
	}

	var res []int
	idx := 0
	for idx < len(expression) {
		for idx < len(expression) && unicode.IsDigit(rune(expression[idx])) {
			idx++
		}
		if idx >= len(expression) {
			break
		}
		left, right := diffWaysToCompute(expression[:idx]), diffWaysToCompute(expression[idx+1:])
		for _, l := range left {
			for _, r := range right {
				switch expression[idx] {
				case '+':
					res = append(res, l+r)
				case '-':
					res = append(res, l-r)
				case '*':
					res = append(res, l*r)
				}

			}
		}
		idx++
	}
	return res
}
```













# 随机数

## [470. 用 Rand7() 实现 Rand10()](https://leetcode.cn/problems/implement-rand10-using-rand7/)

> <img src="./%E6%95%B0%E5%AD%A6-%E5%8D%9A%E5%BC%88-%E8%AE%A1%E7%BB%84-OS%E7%AD%89.assets/image-20230808113419225.png" alt="image-20230808113419225" style="zoom:67%;" />
>
> 两个随机数做乘积表 => Good!

```go
func rand10() int {
	a, b := rand7(), rand7()
	if a > 4 && b < 4 {
		return rand10()
	}
	return (a+b)%10 + 1
}
```

> 先构造大范围的均匀分布
>
> 在获得小的均匀分布

```go
func rand10() int {
	res := 0
	for {
		res = (rand7() - 1) * 7+ rand7() // 构造1-49的均匀分布
		if res <= 40 {
			break
		}
	}
	return res%10 + 1
}
```





# 蓄水池算法

## [398. 随机数索引](https://leetcode.cn/problems/random-pick-index/)

<img src="./%E6%95%B0%E5%AD%A6-%E5%8D%9A%E5%BC%88-%E8%AE%A1%E7%BB%84-OS%E7%AD%89.assets/image-20230808122657085.png" alt="image-20230808122657085" style="zoom:67%;" />

> 如果数组以文件形式存储（读者可假设构造函数传入的是个文件路径），且文件大小远超内存大小，我们是无法通过读文件的方式，将所有下标保存在内存中的，因此需要找到一种空间复杂度更低的算法。
>
> <img src="./%E6%95%B0%E5%AD%A6-%E5%8D%9A%E5%BC%88-%E8%AE%A1%E7%BB%84-OS%E7%AD%89.assets/image-20230808123032367.png" alt="image-20230808123032367" style="zoom:67%;" />
>
> <img src="./%E6%95%B0%E5%AD%A6-%E5%8D%9A%E5%BC%88-%E8%AE%A1%E7%BB%84-OS%E7%AD%89.assets/image-20230808123406982.png" alt="image-20230808123406982" style="zoom:67%;" />

```go
type Solution []int

func Constructor(nums []int) Solution {
	return nums
}

func (nums Solution) Pick(target int) int {
	cnt, res := 0, 0
	for i, num := range nums {
		if num == target {
			cnt++ // 第 cnt 次遇到 target
			if rand.Intn(cnt) == 0 {
				res = i
			}
		}
	}
	return res
}
// 但是会超时哈, 还不如哈希
```

# 博弈

## P2252 取石子游戏|【模板】威佐夫博弈

<img src="./%E6%95%B0%E5%AD%A6-%E5%8D%9A%E5%BC%88-%E8%AE%A1%E7%BB%84-OS%E7%AD%89.assets/image-20230808123845340.png" alt="image-20230808123845340" style="zoom: 80%;" />

> 博弈问题就是要找到 **必胜状态**
>
> 奇异局势 (后手必胜!!!)
>
> - (1, 2) (3, 5) (4, 7)
> - 每个奇异局势的差值都是不同的(为什么呢)
>   - 保证两个奇异局势不能直接一步转化
>   - 但是后手可以直接转化为前面的奇异局势(后手必胜)
> - 两数之差, 分别1,2,3,4,5....
> - 任何一个数对, 无法直接变为(0, 0)
> - 此外的数对,都是可以计数次变为的
> - 所有数对包含了所有的自然数
>
> <img src="./%E6%95%B0%E5%AD%A6-%E5%8D%9A%E5%BC%88-%E8%AE%A1%E7%BB%84-OS%E7%AD%89.assets/image-20230808124718968.png" alt="image-20230808124718968" style="zoom:67%;" />

```go
var F = (1 + math.Sqrt(5)) / 2

func main() {
	var a, b float64
	fmt.Scanln(&a, &b)
	if a > b {
		a, b = b, a
	}
	if int(a/F) == int(b/(F*F)) {
		fmt.Println(0)
	} else {
		fmt.Println(1)
	}
}
```



## [464. 我能赢吗](https://leetcode.cn/problems/can-i-win/)

<img src="./%E6%95%B0%E5%AD%A6-%E5%8D%9A%E5%BC%88-%E8%AE%A1%E7%BB%84-OS%E7%AD%89.assets/image-20230809162113617.png" alt="image-20230809162113617" style="zoom:67%;" />

> 博弈游戏
>
> - 没有规律的时候, 那就是回溯
> - 使用一个dfs同事模拟两个人, 但是不是根据参数区分,而是根据函数体来进行判断
> - 通常要进行记忆化搜索

```go
func canIWin(maxChoosableInteger int, desiredTotal int) bool {
	if (maxChoosableInteger+1)*maxChoosableInteger/2 < desiredTotal {
		return false
	}

	resRecord := map[int]bool{}
	var dfs func(currentSum int, bit int) bool
	dfs = func(currentSum int, bit int) bool {
		res := false // 默认自己不成功
		if can, has := resRecord[bit]; has {
			return can
		}
		for i := 1; i <= maxChoosableInteger; i++ {
			// 这一位还没选
			if bit&(1<<i) == 0 {
				if currentSum+i >= desiredTotal || !dfs(currentSum+i, bit|(1<<i)) {
					res = true // 自己绝对赢了 // 后手绝对赢不了
					break
				}
			}
		}
		resRecord[bit] = res
		return res
	}
	return dfs(0, 0)
}
```



## P8658  填字母游戏

<img src="./%E6%95%B0%E5%AD%A6-%E5%8D%9A%E5%BC%88-%E8%AE%A1%E7%BB%84-OS%E7%AD%89.assets/image-20230810163328774.png" alt="image-20230810163328774" style="zoom:67%;" />



> 注意输入输出
>
> - 这道题不要使用先改数组, 再传入dfs的方法
>
>   因为是提前返回还要进行后面的判断
>
> - 所以要直接再参数里面改

```go
func main() {
	in, out := bufio.NewReader(os.Stdin), bufio.NewWriter(os.Stdout)
	defer out.Flush()
	var n int
	fmt.Fscanf(in, "%d\n", &n)

	var resRecord map[string]int // 记忆化搜索
	var dfs func(s string) int
	dfs = func(s string) int {
		//fmt.Println(s, resRecord)
		if res, has := resRecord[s]; has {
			return res // 记忆化搜索
		}

		if strings.Contains(s, "LOL") {
			resRecord[s] = -1 // 已经被上一家拿下了
			return -1
		}
		if !strings.Contains(s, "*") {
			resRecord[s] = 0 // 平局, 都拿不下
			return 0
		}

		res := -1 // 记录结果, 默认自己输
		for idx := 0; idx < len(s); idx++ {
			if s[idx] == '*' {
				for _, c := range []byte{'O', 'L'} {
					t := dfs(s[:idx] + string(c) + s[idx+1:])
					if t == -1 { // 对手必输
						resRecord[s] = 1
						return 1
					} else if t == 0 {
						res = 0 // 可以平局
					}
				}
			}
		}
		resRecord[s] = res
		return res
	}

	var s string
	for i := 0; i < n; i++ {
		resRecord = map[string]int{}
		fmt.Fscanf(in, "%s\n", &s)
		fmt.Println(dfs(s))
	}

}
```



## [877. 石子游戏](https://leetcode.cn/problems/stone-game/)

<img src="./%E6%95%B0%E5%AD%A6-%E5%8D%9A%E5%BC%88-%E8%AE%A1%E7%BB%84-OS%E7%AD%89.assets/image-20230829225305733.png" alt="image-20230829225305733" style="zoom:67%;" />

> 长度500, 回溯不行, 直接dp
>
> - 直接就是经典的区间dp呀
> - 比的是谁大,所以肯定dp的是差值(可以是作为一种分数)
> - dp都最好留出两侧的端点
> - 而且要注意切换先后手!!!!!!!!!!!!!
> - 代码中的piles[i-1]是自己的分数, dp是之前人的分数, 肯定要选择自己最大的分数呀
> - 找到转义就ok啦
>
> <img src="./%E6%95%B0%E5%AD%A6-%E5%8D%9A%E5%BC%88-%E8%AE%A1%E7%BB%84-OS%E7%AD%89.assets/image-20230829230306430.png" alt="image-20230829230306430" style="zoom:50%;" />

```go
func stoneGame(piles []int) bool {
	n := len(piles)
	dp := make([][]int, n+2)
	for i := 0; i < n+2; i++ {
		dp[i] = make([]int, n+2)
	}

	for length := 1; length <= n; length++ {
		for l := 1; l+length-1 <= n; l++ {
			r := l + length - 1
			dp[l][r] = max(piles[l-1]-dp[l+1][r], piles[r-1]-dp[l][r-1])
		}
	}
	return dp[1][n] > 0
}
```





## [1140. 石子游戏 II](https://leetcode.cn/problems/stone-game-ii/)

<img src="./%E6%95%B0%E5%AD%A6-%E5%8D%9A%E5%BC%88-%E8%AE%A1%E7%BB%84-OS%E7%AD%89.assets/image-20230829232304704.png" alt="image-20230829232304704" style="zoom:67%;" />

> 长度仅为100, 而且递归相对简单
>
> - 可以尝试记忆化搜索
> - 从一个方向拿, 可以前缀和
> - 也是要注意先后手
>
> ![image-20230829233825854](./%E6%95%B0%E5%AD%A6-%E5%8D%9A%E5%BC%88-%E8%AE%A1%E7%BB%84-OS%E7%AD%89.assets/image-20230829233825854.png)

```go
func stoneGameII(piles []int) int {
	prefixSum := make([]int, len(piles)+1)
	for i, v := range piles {
		prefixSum[i+1] = prefixSum[i] + v
	}

	type pair struct{ i, m int }
	dp := map[pair]int{}

	var dfs func(int, int) int
	dfs = func(i int, m int) int {
		if i == len(piles) { // 终止条件
			return 0
		}
		if v, ok := dp[pair{i, m}]; ok { // 记忆化
			return v
		}

		maxVal := math.MinInt
		for x := 1; x <= 2*m; x++ {
			if i+x > len(piles) {
				break
			}
			maxVal = max(maxVal, prefixSum[i+x]-prefixSum[i]-dfs(i+x, max(m, x)))
		}
		dp[pair{i, m}] = maxVal
		return maxVal
	}
	return (prefixSum[len(piles)] + dfs(0, 1)) / 2
}
```

## [810. 黑板异或游戏](https://leetcode.cn/problems/chalkboard-xor-game/)

<img src="./%E6%95%B0%E5%AD%A6-%E5%8D%9A%E5%BC%88-%E8%AE%A1%E7%BB%84-OS%E7%AD%89.assets/image-20230830160405314.png" alt="image-20230830160405314" style="zoom:67%;" />



> - <img src="./%E6%95%B0%E5%AD%A6-%E5%8D%9A%E5%BC%88-%E8%AE%A1%E7%BB%84-OS%E7%AD%89.assets/image-20230830160456303.png" alt="image-20230830160456303" style="zoom:50%;" />
> - <img src="./%E6%95%B0%E5%AD%A6-%E5%8D%9A%E5%BC%88-%E8%AE%A1%E7%BB%84-OS%E7%AD%89.assets/image-20230830160822800.png" alt="image-20230830160822800" style="zoom:50%;" />

```go
func xorGame(nums []int) bool {
    if len(nums)%2 == 0 {
        return true
    }
    xor := 0
    for _, num := range nums {
        xor ^= num
    }
    return xor == 0
}
```





# 几何

## [836/85. 矩形重叠](https://leetcode.cn/problems/rectangle-overlap/)

<img src="./%E6%95%B0%E5%AD%A6-%E5%8D%9A%E5%BC%88-%E8%AE%A1%E7%BB%84-OS%E7%AD%89.assets/image-20231201143641146.png" alt="image-20231201143641146" style="zoom:67%;" />

> - 最开始思考的是至少有一个点在另一个图形里面, 显然是错的
> - 正确思路
>   - x轴一定是有重合部分 and y轴一定是有重合部分的
>   - 可以理解为两个轴上的投影一定有重叠部分

```go
func isRectangleOverlap(rec1 []int, rec2 []int) bool {
	x1, x2, x3, x4 := rec1[0], rec1[2], rec2[0], rec2[2]
	y1, y2, y3, y4 := rec1[1], rec1[3], rec2[1], rec2[3]
	return !(x1 >= x4 || x2 <= x3 || y1 >= y4 || y2 <= y3)
}
```





# 线段树

## 线段树

<img src="./%E6%95%B0%E5%AD%A6-%E5%8D%9A%E5%BC%88-%E8%AE%A1%E7%BB%84-OS%E7%AD%89.assets/image-20231206112305956.png" alt="image-20231206112305956" style="zoom:67%;" />

### 线段树的存储方式

对于区间最值问题, 线段树每个节点存储三个域(左,右,最值)
除了最后一层, 其它层构成一颗满二叉树, 因此采用顺序存储方式, 用一个数组tree存储节点

<img src="./%E6%95%B0%E5%AD%A6-%E5%8D%9A%E5%BC%88-%E8%AE%A1%E7%BB%84-OS%E7%AD%89.assets/image-20231206112841967.png" alt="image-20231206112841967" style="zoom: 50%;" />



### 为什么要开4n空间?

<img src="./%E6%95%B0%E5%AD%A6-%E5%8D%9A%E5%BC%88-%E8%AE%A1%E7%BB%84-OS%E7%AD%89.assets/image-20231206113705342.png" alt="image-20231206113705342" style="zoom:50%;" />



### 创建线段树

递归方式创建

- 叶子结点 => 节点最值就是对应的元素值
- 非叶子结点, 递归创建左右子树
- 区间最值等于左右子树的最值

### 点更新

步骤

- 如果是叶子结点, 满足l=r l=i 更新该节点的值
- 飞叶子结点, 看是左子树还是右子树
- 返回更新节点的最值 // 要向上更新

### 区间查询

- 节点所在区间被[l,r]覆盖, 返回该节点的最值
- 判断是左子树还是右子树
- 返回最值

### 区间修改 + lazy tag

如果修改区间完全覆盖节点区间

- 在节点上修改该区间的值, 打上lazy tag 立即返回
- 下次需要的时候, 再下传lazy tag
- 修改与查询时间控制在 $O(logn)$

区间修改分类之前 => PushDown, 先分账

区间修改完成 => PushDown, 向爹汇报

### 算法分析

<img src="./%E6%95%B0%E5%AD%A6-%E5%8D%9A%E5%BC%88-%E8%AE%A1%E7%BB%84-OS%E7%AD%89.assets/image-20231206121308628.png" alt="image-20231206121308628" style="zoom:67%;" />

## P3372 【模板】线段树 1

<img src="./%E6%95%B0%E5%AD%A6-%E5%8D%9A%E5%BC%88-%E8%AE%A1%E7%BB%84-OS%E7%AD%89.assets/image-20231206160617723.png" alt="image-20231206160617723" style="zoom:67%;" />

> 具体解释详见上一节
>
> - `fmt.Fscanf(in, "%d", &n)`读取数字, 若遇到换行,则必须`fmt.Fscanln(in)`空读一次, 否则会错误
> - 创建,查询,更新的k都是1, 不是1的k知识在里面递归用的,用户看不到的

```go
type node struct {
	// 线段树的节点信息
	// l, r 节点区间
	// value 当前的统计值
	// tag 懒标记
	l, r, value, tag int
}

type Tree struct {
	arr []*node
}

func NewTree(n int) *Tree {
	// n 节点个数, 开4n空间
	return &Tree{
		arr: make([]*node, 4*n),
	}
}

func (t *Tree) pushUp(k int) {
	// 向上更新
	t.arr[k].value = t.arr[k*2].value + t.arr[k*2+1].value
}

func (t *Tree) pushDown(k int) {
	// 向下更新
	if t.arr[k].tag != 0 {
		t.arr[k*2].value += t.arr[k].tag * (t.arr[k*2].r - t.arr[k*2].l + 1)
		t.arr[k*2+1].value += t.arr[k].tag * (t.arr[k*2+1].r - t.arr[k*2+1].l + 1)
		t.arr[k*2].tag += t.arr[k].tag // 现在是儿子们欠孙子们账了
		t.arr[k*2+1].tag += t.arr[k].tag
		t.arr[k].tag = 0 // 父亲的账还完了
	}
}

func (t *Tree) build(k, l, r int, nums []int) {
	// idx真实存储下标, l,r 区间端点
	// nums 原始数组
	t.arr[k] = &node{
		l:   l,
		r:   r,
		tag: 0,
	}
	// 叶子结束递归, 存储最值
	if l == r {
		t.arr[k].value = nums[l-1]
		return
	}
	// 不是叶子 递归
	mid := (l + r) / 2
	t.build(2*k, l, mid, nums)
	t.build(2*k+1, mid+1, r, nums)
	t.pushUp(k) // 左右孩子的值交给父亲
}

func (t *Tree) update(k, l, r, v int) {
	// 使用tag进行区间修改
	// k 真是存储的下标
	// l, r 区间
	// v 修改的值
	if l <= t.arr[k].l && r >= t.arr[k].r {
		// 此节点北区间完全覆盖
		t.arr[k].value += (t.arr[k].r - t.arr[k].l + 1) * v // 欠每个儿子一个v
		t.arr[k].tag += v                                   // 打上标记
		return
	}

	mid := (t.arr[k].l + t.arr[k].r) / 2
	// 必须向儿子们分账
	t.pushDown(k)
	if l <= mid {
		t.update(k*2, l, r, v)
	}
	if r > mid {
		t.update(k*2+1, l, r, v)
	}

	// 再向上 左右儿子必须都找完才会向上
	t.pushUp(k) // 他爹存着他们的value, 所以必须向上
}

func (t *Tree) query(k, l, r int) int {
	if t.arr[k].l >= l && t.arr[k].r <= r {
		// 区间覆盖法
		return t.arr[k].value
	}

	// 开始收缩
	mid := (t.arr[k].l + t.arr[k].r) / 2
	// 向下
	t.pushDown(k)
	// 临时变量
	tmpSum := 0
	if l <= mid {
		// 在左区间去找
		tmpSum += t.query(k*2, l, r)
	}
	if r > mid {
		// 在右区间去找
		tmpSum += t.query(k*2+1, l, r)
	}
	return tmpSum
}
```

acm模式代码

```go
func main() {
	var n, m int
	in := bufio.NewReader(os.Stdin)
	out := bufio.NewWriter(os.Stdout)
	defer out.Flush()

	fmt.Fscanf(in, "%d", &n)
	fmt.Fscanf(in, "%d", &m)
	fmt.Fscanln(in)

	// 创建树
	t := NewTree(n)
	nums := make([]int, n)

	for i := 0; i < n; i++ {
		var tmp int
		fmt.Fscanf(in, "%d", &tmp)
		nums[i] = tmp
	}
	t.build(1, 1, n, nums)
	fmt.Fscanln(in)

	// 读取命令
	for i := 0; i < m; i++ {
		var tmp int
		var x, y, z int
		fmt.Fscanf(in, "%d", &tmp)
		//fmt.Println("tmp", tmp)
		switch tmp {
		case 1:
			fmt.Fscanf(in, "%d", &x)
			fmt.Fscanf(in, "%d", &y)
			fmt.Fscanf(in, "%d", &z)
			fmt.Fscanln(in)
			t.update(1, x, y, z)
		case 2:
			fmt.Fscanf(in, "%d", &x)
			fmt.Fscanf(in, "%d", &y)
			fmt.Fscanln(in)
			fmt.Fprintf(out, "%d\n", t.query(1, x, y))
		}
	}
}
```



# 扫描线算法

## 扫描线

<img src="./%E6%95%B0%E5%AD%A6-%E5%8D%9A%E5%BC%88-%E8%AE%A1%E7%BB%84-OS%E7%AD%89.assets/image-20231207112734973.png" alt="image-20231207112734973" style="zoom:50%;" />

- `2n-1`个区块
- 离散化线段树, 因此要有映射关系!!!!!, 防止区间过大

<img src="./%E6%95%B0%E5%AD%A6-%E5%8D%9A%E5%BC%88-%E8%AE%A1%E7%BB%84-OS%E7%AD%89.assets/image-20231212161839302.png" alt="image-20231212161839302" style="zoom:50%;" />

- 离散化三部曲
  - 排序
  - 去重
  - 二分查找下标
- 如果区间一一映射, 就会留下空挡, 那么怎么办呢?
  - 因为传统用的线段树就是离散的点
  - 但是扫描线是连续的几何
- 其实Tree每个儿子节点对应的就是相邻两个x之间的这一段!!!!!!!!
  - 所以可以使用cnt+tag看覆盖了多少次
  - 如果一个节点自己被覆盖, 就代表两个儿子至少都被覆盖一次,否则分别计算儿子的和
- 真正的计算逻辑就是没两个平行线之间的一定是矩形(可能是多个, 但是都是底乘高)
  - 这样就可以啦
  - 一共计算逻辑上`2*n-1`次

<img src="./%E6%95%B0%E5%AD%A6-%E5%8D%9A%E5%BC%88-%E8%AE%A1%E7%BB%84-OS%E7%AD%89.assets/image-20231212171024011.png" alt="image-20231212171024011" style="zoom:50%;" />

<img src="./%E6%95%B0%E5%AD%A6-%E5%8D%9A%E5%BC%88-%E8%AE%A1%E7%BB%84-OS%E7%AD%89.assets/image-20231212172852947.png" alt="image-20231212172852947" style="zoom:50%;" />

```go
type line struct { // 存储的是一条线
	x1, x2, y int64 // 两横一纵
	tag       int64 // 入边:+1, 出边:-1
}

type node struct {
	// 线段树的节点信息
	// l, r 节点区间
	// cnt 区间覆盖次数, 为正代表这个区间已经被完整覆盖
	// len 区间覆盖长度(它对应的长度里面有多少是被覆盖的)
	// cnt > 0, 则len就是对应的长度
	l, r, cnt, len int64
}

type tree struct {
	nodes []node  // 线段树的所有节点
	lines []line  // 所有线段
	x     []int64 //所有的x
}

func (t *tree) init(n int) {
	// 初始化树, 注意节点数是8倍的点数, 自然就是16倍的矩形数
	t.nodes = make([]node, 2*8*n+1)
	t.lines = make([]line, 2*n+1)
	t.x = make([]int64, 2*n+1)
}

func (t *tree) build(k, l, r int64) {
	// k 真实存储下标, l,r 区间端点
	// 递归向下把l和r填上
	t.nodes[k] = node{l, r, 0, 0}
	if l == r {
		return
	}
	mid := (l + r) / 2
	t.build(2*k, l, mid)
	t.build(2*k+1, mid+1, r)
}

func (t *tree) pushUp(k int64) {
	// 是否区间需要向右边伸展
	l, r := t.nodes[k].l, t.nodes[k].r
	// 若已被完全覆盖,则长度就是节点完全对应的
	if t.nodes[k].cnt != 0 {
		t.nodes[k].len = t.x[r+1] - t.x[l]
	} else {
		// 否则就是两个儿子的
		t.nodes[k].len = t.nodes[2*k].len + t.nodes[2*k+1].len
	}
}

func (t *tree) modify(k, l, r, tag int64) {
	if l > t.nodes[k].r || r < t.nodes[k].l {
		return // 越界
	}
	if l <= t.nodes[k].l && r >= t.nodes[k].r { // 完全覆盖
		t.nodes[k].cnt += tag // 修改一个tag
		t.pushUp(k)           // 此节点要改一下len
		return
	}
	// 已经被完全覆盖就不用向下修改了
	// 否则就是向下
	t.modify(2*k, l, r, tag)
	t.modify(2*k+1, l, r, tag)
	t.pushUp(k)
}

func (t *tree) lowerBound(target int64) int64 {
	// 二分查找x
	// 这样更快啊
	low, high := 0, len(t.x)

	for low < high {
		mid := low + (high-low)/2
		if t.x[mid] < target {
			low = mid + 1
		} else {
			high = mid
		}
	}
	return int64(low)
}

func (t *tree) uniqueX() {
	// 获得x的独有
	// 第一个元素不用, 不要涉及
	if len(t.x) <= 1 {
		return
	}

	i := 1
	for j := 2; j < len(t.x); j++ {
		if t.x[j] != t.x[i] {
			i++
			t.x[i] = t.x[j]
		}
	}
	t.x = t.x[:i+1]
}

func (t *tree) area() int64 {
	// 对X坐标进行排序 并去重
	sort.Slice(t.x[1:], func(i, j int) bool {
		return t.x[1+i] < t.x[1+j] // fkfkfkf 要+1
	})
	t.uniqueX()

	// 对扫描线进行排序
	sort.Slice(t.lines[1:], func(i, j int) bool {
		return t.lines[1+i].y < t.lines[1+j].y
	})

	// 创建树
	t.build(1, 1, int64(len(t.x)))

	// 计算结果
	var res int64 = 0
	for i := 1; i < len(t.lines)-1; i++ {
		l := t.lowerBound(t.lines[i].x1)
		r := t.lowerBound(t.lines[i].x2)
		// 传参是确保修改的区间
		// 所以传参的r要减去1
		t.modify(1, l, r-1, t.lines[i].tag)

		// 到t.lines[i+1].y这一段永远是逻辑上的矩形
		res = res + t.nodes[1].len*(t.lines[i+1].y-t.lines[i].y)
	}
	return res
}

func main() {
	// 输入与输出
	in := bufio.NewReader(os.Stdin)

	// 读取矩形个数
	var n int
	fmt.Fscanln(in, &n)

	// 一颗树
	var t tree
	t.init(n)

	// 读取每个矩形的信息
	for i := 1; i <= n; i++ {
		var x1, y1, x2, y2 int64
		fmt.Fscanln(in, &x1, &y1, &x2, &y2)
		t.lines[i] = line{x1, x2, y1, 1}
		t.lines[i+n] = line{x1, x2, y2, -1}
		t.x[i] = x1
		t.x[i+n] = x2
	}
	fmt.Println(t.area())
}
```



## [850/88. 矩形面积 II](https://leetcode.cn/problems/rectangle-area-ii/)

<img src="./%E6%95%B0%E5%AD%A6-%E5%8D%9A%E5%BC%88-%E8%AE%A1%E7%BB%84-OS%E7%AD%89.assets/image-20231206105344661.png" alt="image-20231206105344661" style="zoom:80%;" />

> 扫描线算法模板题 !!!!!!























