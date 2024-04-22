#Opening db for python
import chess
import chess.polyglot
import time


board = chess.Board()

with chess.polyglot.open_reader("./Books/books/varied.bin") as reader:
   while True:
    try:
      moves = list(reader.find_all(board))
      print(moves[0].move)
      board.push(moves[0].move)
      print(board)
      time.sleep(1)
    except:
      print('No more moves')
      break
     