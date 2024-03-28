#Opening db for python
import chess
import chess.polyglot
import time

board = chess.Board()

with chess.polyglot.open_reader("./Books/books/varied.bin") as reader:

   for i in range(10):
    board.push(list(reader.find_all(board))[0].move)
    print(board)
    time.sleep(1)
     