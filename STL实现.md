# 字符串转浮点数

```go
func stof(s string) (float64, error) {
	// 正负号
	pos := 1

	switch s[0] {
	case '+':
		s = s[1:]
	case '-':
		pos = -1
		s = s[1:]
	case '.':
		return stof("0" + s)
	}

	// 小数点个数, 和小数点后的位数, int64的记录值
	point, next_point := 0, 0
	var ret int64 = 0

	// 遍历s
	for i := 0; i < len(s); i++ {
		// 是小数点
		// 不是数字
		if s[i] == '.' {
			if point != 0 {
				return 0, errors.New("too many points")
			} else {
				point += 1
			}
		} else if s[i] > '9' || s[i] < '0' {
			return 0, errors.New("format error")
		} else {
			ret = ret*10 + int64(int(s[i])-int('0'))
			if point != 0 {
				next_point += 1
			}
		}
	}
	return float64(pos) * float64(ret) / math.Pow10(next_point), nil
}
```

