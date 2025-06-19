board = [
    [5, 3, 0, 0, 7, 0, 0, 0, 0],
    [0, 0, 0, 0, 0, 0, 0, 0, 0],  # 修改了其他行，避免重复
    [0, 0, 0, 0, 0, 0, 0, 0, 0],
    [0, 0, 0, 0, 0, 0, 0, 0, 0],
    [0, 0, 0, 0, 0, 0, 0, 0, 0],
    [0, 0, 0, 0, 0, 0, 0, 0, 0],
    [0, 0, 0, 2, 0, 0, 0, 0, 0],
    [0, 0, 0, 0, 0, 0, 0, 0, 0],
    [0, 0, 0, 0, 0, 0, 0, 0, 0]
]

def printboard(board, given):
    for i in range(9):
        print("-------------------------------------")
        print(end="| ")
        for j in range(9):
            num = board[i][j]
            if num == 0:
                print("·", end=" | ")
            elif given[i][j]:
                print(f"\033[95m{num}\033[0m", end=" | ")  # 題目：藍色
            else:
                print(f"\033[96m{num}\033[0m", end=" | ")  # 玩家填入：綠色
        print()

def is_valid(board, row, col, num):
    # Check if the cell is empty
    if board[row][col] != 0:
        return False # Cell is empty, so placing 'num' is valid
    if num in [board[row][i] for i in range(9)]:
        return False
    if num in [board[i][col] for i in range(9)]:
        return False
    startR, startC = 3*(row//3),3*(col//3)
    for i in range(3):
        for j in range(3):
            if board[startR+i][startC+j] == num:
                return False
    return True

def solve(board):
    for row in range(9):
        for col in range(9):
            if board[row][col] == 0:
                for num in range(1,10):
                    if is_valid(board, row, col, num):
                        board[row][col] = num
                        if solve(board):
                            return True
                        board[row][col] = 0
                return False
    return True

if __name__ == "__main__":
    given = [[board[i][j] != 0 for j in range(9)] for i in range(9)]
    print("原始題目：")
    printboard(board, given)
    while True:
        user_input = input("輸入數字 (1-9) 或輸入 s 解題 / q 離開：")
        if user_input.lower() == "q":  
            print("退出遊戲！")
            break
        if user_input.lower() == "s":
            if solve(board):
                print("AI 解題成功！")
            else:
                print("這題無解。")
            printboard(board, given)
            continue    
        try:
            a = int(user_input)
            r = int(input("輸入行 (0-8): "))
            c = int(input("輸入列 (0-8): "))
            
            if not (1 <= a <= 9 and 0 <= r <= 8 and 0 <= c <= 8):
                print("輸入無效！請重新輸入。")
                continue
            
            if is_valid(board, r, c, a):
                board[r][c] = a
                print("填入成功！")
            else:
                print("無效的填入！請檢查數字是否衝突。")
            
            printboard(board)
        except ValueError:
            print("請輸入數字！")
