#-*-coding:utf-8-*-

def get_h(lon, lat, lst):
    p0 = [lon, lat]
    sum0 = 0
    sum1 = 0
    temp = []
    # 遍历获取该点距离所有采样点的距离
    for point in lst:
        if lon == point[0] and lat == point[1]:
            return point[2]
        Di = distance(p0, point)
        # new出来一个对象，不然会改变原来lst的值
        ptn = copy.deepcopy(point)
        ptn.append(Di)
        temp.append(ptn)

    # 根据上面ptn.append（）的值由小到大排序
    temp1 = sorted(temp, key=lambda point: point[3])
    # 遍历排序的前15个点，根据公式求出sum0 and sum1
    for point in temp1[0:15]:
        sum0 += point[2] / point[3]
        sum1 += 1 / point[3]

    return sum0 / sum1


# 计算两点间的距离
def distance(p, pi):
    dis = (p[0] - pi[0]) * (p[0] - pi[0]) + (p[1] - pi[1]) * (p[1] - pi[1])
    m_result = math.sqrt(dis)
    return m_result

