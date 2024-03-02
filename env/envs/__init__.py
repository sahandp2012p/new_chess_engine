from chess_gym.envs import ChessEnv
import numpy as np
import chess

class MoveSpace:
    def __init__(self, board):
        self.board = board
        self.n = len(list(self.board.legal_moves))

    def sample(self):
        return np.random.choice(list(self.board.legal_moves))
    
class ChessEnv(ChessEnv):
    def __init__(self, chess960):
        super().__init__(chess960=chess960)
        self.reward_lookup = {
            'check': 0.05,
            'mate': 100.0,
            'stalemate': 0.0,
            'p': 1,
            'n': 3,
            'b': 3,
            'r': 5,
            'q': 9,
            '2': 3,  # Promotion to knight
            '3': 3,  # Promotion to bishop
            '4': 5,  # Promotion to rook
            '5': 9   # Promotion to queen
        }

        self.action_space = MoveSpace(self.board)

    def _update_reward(self):
        reward = 0

        if self.board.is_check():
            reward += self.reward_lookup['check']

        end_game_result = self.board.result()
        if '1-0' in end_game_result or '0-1' in end_game_result:
            reward = self.reward_lookup['mate']
        elif '1/2-1/2' in end_game_result:
            reward = self.reward_lookup['stalemate']

        return reward
    
    def _generate_reward(self, action):
        """Assign rewards to moves, captures, queening, checks, and winning"""
        reward = 0.0
        piece_map = self.board.piece_map()

        to_square = action.to_square
        if self.board.is_capture(action):
            captured_piece = piece_map[to_square].symbol()
            reward = self.reward_lookup[captured_piece.lower()]

        promotion = action.promotion
        if promotion is not None:
            reward += self.reward_lookup[str(promotion)]
        return reward

    def step(self, action):
        reward = self._generate_reward(action)
        observation = self._observe()

        self.board.push(action)
        terminal = self.board.is_game_over(claim_draw = self.claim_draw)
        info = {'turn': self.board.turn,
                'castling_rights': self.board.castling_rights,
                'fullmove_number': self.board.fullmove_number,
                'halfmove_clock': self.board.halfmove_clock,
                'promoted': self.board.promoted,
                'chess960': self.board.chess960,
                'ep_square': self.board.ep_square}

        return observation, reward, terminal, info
    