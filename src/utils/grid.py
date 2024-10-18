from typing import *
from typing import List


class Grid:
  @staticmethod
  def unflatten_grid(locs: List[Tuple[int, int]], width: int, height: int) -> List[List[Optional[Tuple[int, int]]]]:
    grid_2d: List[List[Optional[Tuple[int, int]]]] = [[None for _ in range(width)] for _ in range(height)]

    # for row in grid_2d:
    #   print(row)

    for x, y in locs:
      try:
        grid_2d[y][x] = (x, y)
      except IndexError:
        return [[None] * width] * height

    return grid_2d

  @staticmethod
  def render_grid(grid: List[List[Optional[Tuple[int, int]]]]) -> list[str]:
    max_lengths = [0] * len(grid[0])

    for row in grid:
      cleaned_row = [cell for cell in row if cell is not None]

      for i, cell in enumerate(cleaned_row):
        max_lengths[i] = max(max_lengths[i], len(str(cell)))

    formatted_grid = []
    for row in grid:
      formatted_row = []
      for i, cell in enumerate(row):
        # if cell is None:
        #   formatted_row.append(" " * (max_lengths[i] + 1) + "|")
        # else:
        #   s = str(cell).rjust(max_lengths[i]) + "|"
        #   if i == 0:
        #     formatted_row.append("|" + s)
        #   else:
        #     formatted_row.append(s)
        if cell is None:
          formatted_row.append(" " * (max_lengths[i]))
        else:
          s = str(cell).rjust(max_lengths[i])
          if i == 0:
            formatted_row.append(s)
          else:
            formatted_row.append(s)

      row_s = "".join(formatted_row)
      formatted_grid.append(row_s)

    return formatted_grid[::-1]


if __name__ == '__main__':
  g = Grid.unflatten_grid([(1, 7), (1, 6), (1, 5), (1, 4), (2, 4), (0, 3), (1, 3), (1, 2), (1, 1), (1, 0)], 3, 8)
  print("\n".join(Grid.render_grid(g)))
