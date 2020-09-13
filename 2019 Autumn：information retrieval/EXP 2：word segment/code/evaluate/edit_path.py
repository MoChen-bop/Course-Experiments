def get_edit_path(gold, segment):
    n = len(gold)
    m = len(segment)
    edit_path = [[[] for j in range(m + 1)] for i in range(n + 1)]
    
    for i in range(n + 1):
        edit_path[i][0] = i
    for j in range(m + 1):
        edit_path[0][j] = j
        
    for i in range(1, n + 1):
        for j in range(1, m + 1):
            if gold[i - 1] == segment[j - 1]:
                edit_path[i][j] = edit_path[i - 1][j - 1]
            else:
                modify_cost = edit_path[i - 1][j - 1] + 1
                delete_cost = edit_path[i - 1][j] + 1
                insert_cost = edit_path[i][j - 1] + 1
                edit_path[i][j] = min([modify_cost, delete_cost, insert_cost])
    return edit_path        


def get_operation(edit_path, gold, segment):
    n = len(edit_path)
    m = len(edit_path[0])
    
    edit_opt = []
    
    i = n - 1
    j = m - 1
    while not i == 0 and not j == 0:
        if i == 0:
            edit_opt.append('D')
            j -= 1
        elif j == 0:
            edit_opt.append('I')
            i -= 1
        elif i >= 1 and j >= 1 and edit_path[i][j] == edit_path[i - 1][j - 1] \
            and gold[i - 1] == segment[j - 1]:
            edit_opt.append('-')
            i -= 1
            j -= 1
        else:
            if i >= 1 and edit_path[i][j] == edit_path[i - 1][j] + 1 :
                edit_opt.append('I')
                i -= 1
            elif j >= 1 and edit_path[i][j] == edit_path[i][j - 1] + 1:
                edit_opt.append('D')
                j -= 1
            else:
                edit_opt.append('M')
                i -= 1
                j -=1
    edit_opt.reverse()
    return edit_opt


def get_split_point(segment):
    segment_point = []
    index = 0
    for word in segment:
        index += len(word)
        segment_point.append(index)
    return segment_point


def compare(gold, segment):
    gold_point = get_split_point(gold)
    segment_point = get_split_point(segment)
    edit_path = get_edit_path(gold_point, segment_point)
    edit_operation = get_operation(edit_path, gold_point, segment_point)
    I_count = 0
    D_count = 0
    M_count = 0
    for operation in edit_operation:
        if operation == 'I':
            I_count += 1
        elif operation == 'D':
            D_count += 1
        elif operation == 'M':
            M_count += 1
    return (I_count, D_count, M_count)