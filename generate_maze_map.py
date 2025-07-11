import numpy as np
from PIL import Image
import random

def create_maze(width=51, height=51):
    # 1. 기본 설정 및 맵 초기화
    WALL = 1
    PATH = 255
    # 홀수 크기로 설정하여 벽을 깔끔하게 만듭니다.
    maze = np.full((height, width), WALL, dtype=np.uint8)
    
    # 2. 탐색 시작점 선택
    start_x, start_y = (random.randint(0, width // 2) * 2 + 1, 
                        random.randint(0, height // 2) * 2 + 1)
    
    maze[start_y, start_x] = PATH
    stack = [(start_x, start_y)]

    # 3. 알고리즘 주 루프
    while stack:
        current_x, current_y = stack[-1]
        neighbors = []

        # 방문하지 않은 이웃 찾기
        for dx, dy in [(0, -2), (0, 2), (-2, 0), (2, 0)]:
            nx, ny = current_x + dx, current_y + dy
            if 0 < nx < width-1 and 0 < ny < height-1 and maze[ny, nx] == WALL:
                neighbors.append((nx, ny))
        
        if neighbors:
            # 갈 수 있는 방향이 있으면
            next_x, next_y = random.choice(neighbors)
            
            # 벽 허물기
            wall_x, wall_y = current_x + (next_x - current_x) // 2, current_y + (next_y - current_y) // 2
            maze[wall_y, wall_x] = PATH
            
            # 다음 위치로 이동
            maze[next_y, next_x] = PATH
            stack.append((next_x, next_y))
        else:
            # 막다른 길이면 역추적
            stack.pop()
            
    return maze

if __name__ == '__main__':
    # 맵 생성
    maze_data = create_maze(width=500, height=500)
    
    # 4. 이미지 파일로 저장
    img = Image.fromarray(maze_data, 'L') # 'L'은 8비트 흑백 모드
    img.save('maze.png')
    print("maze.png 파일이 생성되었습니다.")