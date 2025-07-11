import numpy as np
from PIL import Image
import random
from scipy.ndimage import binary_dilation

def create_simple_maze(width=500, height=500, complexity_ratio=0.3):
    """
    단순한 구조의 미로를 생성하는 함수
    :param width: 최종 이미지의 가로 픽셀 수
    :param height: 최종 이미지의 세로 픽셀 수
    :param complexity_ratio: 미로의 복잡도 (0.1: 매우 단순, 1.0: 매우 복잡)
    """
    WALL = 0
    PATH = 255
    
    # 짝수 크기 호환을 위해 내부적으로 홀수 크기로 작업
    maze_width = width - 1 if width % 2 == 0 else width
    maze_height = height - 1 if height % 2 == 0 else height
    
    maze = np.full((maze_height, maze_width), WALL, dtype=np.uint8)
    
    # 1. 시작점 선택
    start_x, start_y = (random.randint(0, (maze_width-1) // 2 - 1) * 2 + 1, 
                        random.randint(0, (maze_height-1) // 2 - 1) * 2 + 1)
    
    maze[start_y, start_x] = PATH
    stack = [(start_x, start_y)]
    
    # 복잡도 조절을 위한 변수
    total_cells = ((maze_width - 1) // 2) * ((maze_height - 1) // 2)
    target_path_cells = int(total_cells * complexity_ratio)
    carved_cells = 1

    # 2. 알고리즘 주 루프 (스택이 비거나 목표 복잡도에 도달할 때까지)
    while stack and carved_cells < target_path_cells:
        current_x, current_y = stack[-1]
        neighbors = []

        # 방문하지 않은 이웃 찾기
        for dx, dy in [(0, -2), (0, 2), (-2, 0), (2, 0)]:
            nx, ny = current_x + dx, current_y + dy
            if 0 < nx < maze_width-1 and 0 < ny < maze_height-1 and maze[ny, nx] == WALL:
                neighbors.append((nx, ny))
        
        if neighbors:
            next_x, next_y = random.choice(neighbors)
            wall_x, wall_y = current_x + (next_x - current_x) // 2, current_y + (next_y - current_y) // 2
            maze[wall_y, wall_x] = PATH
            maze[next_y, next_y] = PATH
            stack.append((next_x, next_y))
            carved_cells += 1
        else:
            stack.pop()

    # 3. 최종 크기에 맞게 테두리 추가 (요청 크기가 짝수였을 경우)
    final_maze = np.full((height, width), WALL, dtype=np.uint8)
    final_maze[:maze_height, :maze_width] = maze
            
    return final_maze

if __name__ == '__main__':
    # 원하는 맵 크기 설정
    MAP_WIDTH = 500
    MAP_HEIGHT = 500
    
    # 미로의 복잡도 설정 (값이 작을수록 단순해집니다)
    COMPLEXITY = 0.1  # 30%만 길을 뚫어 단순한 구조로 만듦

    # 맵 생성
    maze_data = create_simple_maze(width=MAP_WIDTH, height=MAP_HEIGHT, complexity_ratio=COMPLEXITY)
    
    # --- 팽창(Dilation) 연산 추가 ---
    # 2. '페인트 롤러'의 크기를 정합니다. (3x3 크기는 3픽셀 너비의 길을 만듭니다)
    #    True/False 값으로 연산하기 위해 0보다 큰 값을 True로 변환합니다.
    path_mask = maze_data > 0
    
    # 3. 팽창 연산을 적용하여 길을 넓힙니다.
    #    structure의 크기로 롤러 크기(복도 너비)를 조절할 수 있습니다.
    dilated_mask = binary_dilation(path_mask, structure=np.ones((3, 3)))

    # 4. 최종 맵 데이터 생성
    #    팽창된 마스크를 기반으로, 길(True)은 255로, 벽(False)은 0으로 설정합니다.
    final_maze_data = np.zeros((MAP_HEIGHT, MAP_WIDTH), dtype=np.uint8)
    final_maze_data[dilated_mask] = 255
    # --- 연산 추가 끝 ---
    
    # 5. 넓어진 미로 이미지 파일로 저장
    img = Image.fromarray(final_maze_data, 'L')
    img.save(f'wide_maze_{MAP_WIDTH}x{MAP_HEIGHT}.png')
    print(f"{MAP_WIDTH}x{MAP_HEIGHT} 크기의 wide_maze.png 파일이 생성되었습니다.")