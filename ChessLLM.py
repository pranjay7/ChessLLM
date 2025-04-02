import os
import sys
import chess
import chess.pgn
import google.generativeai as genai
import random
import time
from flask import Flask, request, jsonify, Response
from flask_cors import CORS
from dotenv import load_dotenv
import traceback
import requests
# Import Stockfish with a check
try:
    from stockfish import Stockfish
    STOCKFISH_LIB_AVAILABLE = True
except ImportError:
    STOCKFISH_LIB_AVAILABLE = False
    print("ERROR: 'stockfish' library not found. Please install it: pip install stockfish")

# --- Constants ---
STOCKFISH_THINK_TIME_MS = 3000 # 3 seconds, adjust if needed

# --- Environment & API Key Setup ---
print("--- Loading Environment Variables ---")
load_dotenv() # Load .env file if it exists
print(".env loaded (if present).")

# --- Stockfish Setup ---
STOCKFISH_PATH_FROM_ENV = os.getenv("STOCKFISH_PATH")
stockfish_engine = None

print("\n--- Stockfish Initialization Attempt ---")
print(f"Trying to read STOCKFISH_PATH environment variable...")
print(f"Value read from os.getenv('STOCKFISH_PATH'): '{STOCKFISH_PATH_FROM_ENV}'")

if not STOCKFISH_LIB_AVAILABLE:
    print("ERROR: Stockfish library is not installed. Cannot proceed with Stockfish engine.")
elif not STOCKFISH_PATH_FROM_ENV:
    print("ERROR: STOCKFISH_PATH environment variable is NOT SET or empty.")
    print("       Ensure it's set via OS or in a .env file pointing to the executable.")
else:
    print(f"Checking if path exists: '{STOCKFISH_PATH_FROM_ENV}'")
    path_exists = os.path.exists(STOCKFISH_PATH_FROM_ENV)
    print(f"Result of os.path.exists: {path_exists}")

    if not path_exists:
        print(f"ERROR: The specified path does NOT exist: '{STOCKFISH_PATH_FROM_ENV}'")
        print(f"       Current working directory is: {os.getcwd()}")
    else:
        print(f"Checking if path is a file: '{STOCKFISH_PATH_FROM_ENV}'")
        is_file = os.path.isfile(STOCKFISH_PATH_FROM_ENV)
        print(f"Result of os.path.isfile: {is_file}")

        if not is_file:
            print(f"ERROR: The specified path exists, but it is a DIRECTORY, not a file: '{STOCKFISH_PATH_FROM_ENV}'")
            print(f"       STOCKFISH_PATH must point to the Stockfish *executable* file itself.")
        else:
            if os.name != 'nt':
                print(f"Checking execute permissions (non-Windows): '{STOCKFISH_PATH_FROM_ENV}'")
                has_execute_permission = os.access(STOCKFISH_PATH_FROM_ENV, os.X_OK)
                print(f"Result of os.access(path, os.X_OK): {has_execute_permission}")
                if not has_execute_permission:
                    print(f"WARNING: Execute permission might be missing for: '{STOCKFISH_PATH_FROM_ENV}'. Attempting to continue...")
                    # You might need `chmod +x` externally if initialization fails here

            print("\nAttempting to create Stockfish instance...")
            try:
                CURRENT_STOCKFISH_PATH = STOCKFISH_PATH_FROM_ENV
                initial_parameters = {
                    "Move Overhead": 10,
                    # Add Threads/Hash back if needed and tested:
                    # "Threads": 2,
                    # "Hash": 128,
                }
                print(f"Calling Stockfish(path='{CURRENT_STOCKFISH_PATH}', parameters={initial_parameters})...")
                stockfish_engine = Stockfish(path=CURRENT_STOCKFISH_PATH, parameters=initial_parameters)
                print("Stockfish instance created (process likely started).")

                print("Verifying Stockfish basic communication (is_fen_valid)...")
                is_valid = stockfish_engine.is_fen_valid("rnbqkbnr/pppppppp/8/8/8/8/PPPPPPPP/RNBQKBNR w KQkq - 0 1")
                print(f"Result of is_fen_valid: {is_valid}")

                if is_valid:
                    current_params = stockfish_engine.get_parameters()
                    print(f"SUCCESS: Stockfish Engine Initialized and Validated.")
                    print(f"         Current Parameters reported by engine: {current_params}")
                else:
                    print("ERROR: Stockfish process started but failed basic FEN validation (communication error?).")
                    stockfish_engine = None

            except Exception as e:
                print(f"\n!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!")
                print(f"CRITICAL ERROR during Stockfish() instance creation or validation:")
                print(f"Path used: '{CURRENT_STOCKFISH_PATH}'")
                print(f"Error Type: {type(e).__name__}")
                print(f"Error Message: {e}")
                print(f"Traceback:")
                traceback.print_exc()
                print(f"!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!\n")
                stockfish_engine = None

print("\n--- Final Stockfish Engine State Check ---")
if stockfish_engine:
    print("Status: Stockfish engine object IS initialized.")
else:
    print("Status: Stockfish engine object IS NONE (Initialization failed or was disabled).")
print("--- Stockfish Initialization Attempt Complete ---\n")

# --- Gemini Setup ---
GEMINI_API_KEY = os.getenv("GOOGLE_API_KEY") or os.getenv("CURSOR_GOOGLE_API_KEY")
genai_model = None
MODEL_NAME = 'gemini-1.5-flash-latest'
if GEMINI_API_KEY and genai is not None : # Check if import worked
    try:
        genai.configure(api_key=GEMINI_API_KEY)
        genai_model = genai.GenerativeModel(MODEL_NAME)
        print(f"Gemini Model Initialized ({MODEL_NAME}) for 'Ask Piece'.")
    except Exception as e:
        print(f"Error initializing Gemini model: {e}. 'Ask Piece' feature disabled.")
        genai_model = None # Ensure it's None on error
else:
    if not GEMINI_API_KEY:
        print("Warning: GOOGLE_API_KEY environment variable not set. 'Ask Piece' feature disabled.")
    # No need for else here, genai_model is already None


# --- Flask App Setup ---
app = Flask(__name__, static_folder='static', static_url_path='/static')
CORS(app, resources={r"/api/*": {"origins": "*"}})

# --- Game State ---
board = chess.Board()
game_history_san = []

# --- Helper Functions ---
def format_history_as_pgn_movetext(history_san):
    if not history_san: return "(No moves played yet)"
    pgn_string = ""
    for i, move in enumerate(history_san):
        move_number = (i // 2) + 1
        if i % 2 == 0: pgn_string += f"{move_number}. {move} "
        else: pgn_string += f"{move} "
    return pgn_string.strip()

# --- Stockfish Move Function ---
def get_engine_move(current_board):
    """Gets the best move from the Stockfish engine."""
    if not stockfish_engine:
        print("DEBUG: get_engine_move called, but stockfish_engine is None. Choosing random.")
        legal_moves_san = [current_board.san(m) for m in current_board.legal_moves]
        return random.choice(legal_moves_san) if legal_moves_san else None

    if not current_board.legal_moves:
        print("Error: No legal moves available on board for Stockfish.")
        return None

    fen = current_board.fen()
    print(f"--- Requesting Stockfish Move ---")
    print(f"Board FEN: {fen}")
    print(f"Requested Think Time: {STOCKFISH_THINK_TIME_MS}ms")

    try:
        is_valid = stockfish_engine.is_fen_valid(fen)
        if not is_valid:
            print(f"ERROR: Current FEN '{fen}' is reported as invalid by Stockfish just before search.")
            raise ValueError("Invalid FEN detected before Stockfish search")

        stockfish_engine.set_fen_position(fen)
        print(f"FEN position set successfully.")

        start_time = time.perf_counter()
        print(f"Calling get_best_move_time({STOCKFISH_THINK_TIME_MS})...")
        best_move_uci = stockfish_engine.get_best_move_time(STOCKFISH_THINK_TIME_MS)
        end_time = time.perf_counter()
        elapsed_ms = (end_time - start_time) * 1000
        print(f"get_best_move_time completed.")
        print(f"Actual time elapsed for search call: {elapsed_ms:.2f} ms")

        if not best_move_uci:
            print("WARNING: Stockfish returned no move.")
            if current_board.is_game_over(claim_draw=True):
                 print("Reason: Game is already over.")
                 return None
            else:
                 print("Reason: Unknown (Engine error? Timeout too short?). Choosing random.")
                 legal_moves_san = [current_board.san(m) for m in current_board.legal_moves]
                 return random.choice(legal_moves_san) if legal_moves_san else None
        else:
             print(f"Stockfish raw output (UCI): {best_move_uci}")

        try:
            move = current_board.parse_uci(best_move_uci)
            if move in current_board.legal_moves:
                san_move = current_board.san(move)
                print(f"Stockfish proposed valid move: {best_move_uci} ({san_move})")
                print(f"--- Stockfish Move Complete ---")
                return san_move
            else:
                print(f"CRITICAL ERROR: Stockfish proposed ILLEGAL move: {best_move_uci} for FEN {fen}")
                print(f"Legal moves were: {[m.uci() for m in current_board.legal_moves]}")
                print("Choosing random move as emergency fallback.")
                legal_moves_san = [current_board.san(m) for m in current_board.legal_moves]
                return random.choice(legal_moves_san) if legal_moves_san else None
        except ValueError as e_parse:
            print(f"ERROR: Stockfish returned invalid UCI notation: '{best_move_uci}'. Error: {e_parse}")
            print("Choosing random move due to invalid notation.")
            legal_moves_san = [current_board.san(m) for m in current_board.legal_moves]
            return random.choice(legal_moves_san) if legal_moves_san else None

    except Exception as e:
        print(f"ERROR: Exception during Stockfish communication/search: {e}")
        traceback.print_exc()
        print("Choosing random move due to Stockfish communication error.")
        legal_moves_san = [current_board.san(m) for m in current_board.legal_moves]
        return random.choice(legal_moves_san) if legal_moves_san else None


# --- Gemini Piece Perspective Function ---
def get_llm_piece_perspective(fen, history_san, piece_code, square, question):
    """Gets perspective from Gemini API (if available)."""
    if not genai_model:
        print("DEBUG: get_llm_piece_perspective called, but genai_model is None.")
        return "'Ask Piece' feature unavailable (Gemini not configured)."

    # Check if piece code is valid before proceeding
    if not (isinstance(piece_code, str) and len(piece_code) == 2 and piece_code[1].upper() in 'PNBRQK'):
         print(f"Warning: Invalid piece code '{piece_code}' passed to get_llm_piece_perspective.")
         return "Internal error: Invalid piece specified."

    piece_map = {'P':'Pawn','N':'Knight','B':'Bishop','R':'Rook','Q':'Queen','K':'King'}
    piece_type = piece_map.get(piece_code[1].upper(), 'Unknown Piece')
    pgn_movetext = format_history_as_pgn_movetext(history_san)

    # Simple validation for square and FEN before making API call
    try:
        chess.parse_square(square)
        chess.Board(fen) # Basic FEN syntax check
    except ValueError as e:
        print(f"Error: Invalid square ('{square}') or FEN provided to get_llm_piece_perspective: {e}")
        return "Internal error: Invalid board state for query."

    prompt = f"""You ARE the White {piece_type} currently located on square {square}.
The current board state (FEN) is: {fen}.
The game history (PGN) is: {pgn_movetext}.

Answer the user's question strictly from your own perspective as this specific piece on {square}. Observe your thoughts. Think in steps, contemplate on your thoughts and the position for a while, then answer. Provide a compact answer.:
Focus ONLY on:
- Your own safety: Are you currently under attack? By which pieces?
- Squares you attack or defend: Which squares can you move to? Which squares do you protect?
- Threats you pose: Are you attacking any opponent pieces directly?
- Your own possible legal moves: List 1-3 of your possible moves from {square}. Briefly mention immediate consequences like captures or checks *you* would deliver.
- Your direct involvement in the history: Mention if you captured a piece, were recently attacked, or delivered a check, if relevant to the question.

**Crucially, DO NOT:**
- Suggest moves for *any other* White piece besides yourself.
- Give general strategic advice about the overall position or White's plan.
- Analyze the game from the perspective of the player or an external observer.
- Discuss the 'best' move overall for White; only discuss *your* potential moves from {square}.
- Talk about what other pieces *should* do.

Answer in the first person ('I', 'me', 'my'). Be concise and directly address the user's question.
User asks: "{question}"

Your response (as the {piece_type} on {square}):"""

    print(f"Requesting LLM piece perspective ({piece_type}@{square})...")
    try:
        gen_config = genai.types.GenerationConfig(max_output_tokens=150, temperature=0.6)
        response = genai_model.generate_content(prompt, generation_config=gen_config)

        # Basic safety check on response
        if not hasattr(response, 'text') or not isinstance(response.text, str):
             print(f"Warning: Unexpected response format from Gemini API: {response}")
             return "Received an unexpected response from the AI."

        answer = response.text.strip()
        # ... (prefix removal logic remains same) ...
        prefixes_to_remove = [
            f"As the {piece_type} on {square},", f"Okay, speaking as the {piece_type} on {square}:",
            "Okay, here's my perspective:", "Alright, here's what I see:",
            f"I am the {piece_type} on {square}.", "My response (as the",
        ]
        original_answer = answer
        for prefix in prefixes_to_remove:
            # Case-insensitive prefix check
            if answer.lower().startswith(prefix.lower()):
                answer = answer[len(prefix):].lstrip(" ,:.-")
                break # Remove only the first matching prefix
        print(f"LLM Raw Piece Response: '{original_answer}'")
        print(f"LLM Cleaned Piece Response: '{answer}'")
        return answer if answer else "I have no thoughts on that right now."

    except Exception as e:
        print(f"ERROR: Gemini piece perspective API call failed: {e}")
        traceback.print_exc()
        return "My thoughts are clouded... (AI API Error)"


# --- API Endpoints ---
@app.route('/api/game', methods=['GET'])
def get_game_state():
    """Returns the current game state."""
    global board, game_history_san
    winner, termination = None, None
    is_game_over = board.is_game_over(claim_draw=True)
    if is_game_over:
        outcome = board.outcome(claim_draw=True);
        if outcome:
            winner = "white" if outcome.winner == chess.WHITE else "black" if outcome.winner == chess.BLACK else "draw"
            termination = outcome.termination.name.capitalize().replace('_', ' ')
    # Ensure history is sync'd (simple check)
    if not game_history_san and board.move_stack:
       print("Warning: Game history empty but move stack exists. State inconsistency possible.")
       # Could attempt to rebuild history here, but safer to just report state
    return jsonify({
        'fen': board.fen(),
        'turn': 'white' if board.turn == chess.WHITE else 'black',
        'is_game_over': is_game_over,
        'winner': winner,
        'termination': termination,
        'history': game_history_san
    })

@app.route('/api/move', methods=['POST'])
def make_move():
    """Processes the user's move and gets the opponent's response."""
    global board, game_history_san
    if board.turn != chess.WHITE:
        return jsonify({'error': 'Not White\'s turn'}), 400
    if board.is_game_over(claim_draw=True):
        return jsonify({'error': 'Game is over'}), 400

    data = request.get_json()
    uci_move_str = data.get('move')
    if not uci_move_str or not (4 <= len(uci_move_str) <= 5): # Basic UCI validation
        return jsonify({'error': f'Invalid or missing UCI move string: {uci_move_str}'}), 400

    opponent_san_response, move_source = None, None
    try:
        # --- Process User Move ---
        move = board.parse_uci(uci_move_str)
        if move in board.legal_moves:
            san_move = board.san(move)
            board.push(move)
            game_history_san.append(san_move)
            print(f"User (W): {uci_move_str} ({san_move}) | Ply: {len(game_history_san)}")

            # --- Get Opponent Move (if game not over) ---
            if not board.is_game_over(claim_draw=True):
                opponent_san_response, move_source = get_engine_move(board), "Stockfish"

                # Fallback if engine is None or fails
                if opponent_san_response is None:
                     is_engine_disabled = (stockfish_engine is None) # Check if it failed because engine is off
                     fallback_reason = "Engine disabled" if is_engine_disabled else "Engine move generation failed"
                     print(f"{fallback_reason}. Attempting random move.")
                     move_source = "Random(Fallback)"
                     legal_moves = [board.san(m) for m in board.legal_moves]
                     if legal_moves:
                         opponent_san_response = random.choice(legal_moves)
                     else:
                         print("Error: No legal moves available for opponent fallback.")
                         opponent_san_response = None # Ensure it's None if no fallback possible

                # --- Process Opponent Move ---
                if opponent_san_response:
                    try:
                        opp_move = board.parse_san(opponent_san_response)
                        if opp_move in board.legal_moves:
                            board.push(opp_move)
                            game_history_san.append(opponent_san_response)
                            print(f"Opponent ({move_source}): {opponent_san_response}")
                        else:
                            # This is a serious error if the engine/random proposed it
                            print(f"CRITICAL ERROR: Opponent ({move_source}) proposed illegal move {opponent_san_response}. Board FEN: {board.fen()}")
                            # What to do here? Maybe don't push, return error? For now, just log.
                    except ValueError as e_parse:
                        print(f"ERROR: Parsing opponent ({move_source}) SAN '{opponent_san_response}' failed: {e_parse}")
                    except Exception as e_push:
                        print(f"ERROR: Pushing opponent ({move_source}) move '{opponent_san_response}' failed: {e_push}")
                else:
                    # This happens if even random fallback found no moves (stalemate/checkmate just occurred?)
                    print(f"Opponent ({move_source}) could not generate a move (likely game end).")

            # --- Return Final State ---
            final_state_data = get_game_state().get_json() # Use correct variable name
            final_state_data['last_opponent_move'] = opponent_san_response
            return jsonify(final_state_data)

        else:
            # User's move was illegal
            try:
                illegal_san = board.san(move) # Try to get SAN for clarity
            except ValueError:
                illegal_san = uci_move_str # Fallback to UCI if SAN fails
            print(f"User attempted illegal move: {illegal_san}")
            return jsonify({'error': f'Illegal move: {illegal_san}'}), 400

    except ValueError as e_uci:
        # Error parsing user's UCI string
        print(f"Error parsing user UCI '{uci_move_str}': {e_uci}")
        return jsonify({'error': f'Invalid move notation: {uci_move_str}'}), 400
    except Exception as e:
        # Catch-all for unexpected errors during move processing
        print(f"FATAL Error during move processing: {e}")
        traceback.print_exc()
        return jsonify({'error': 'Server error during move processing'}), 500


@app.route('/api/reset', methods=['POST'])
def reset_game():
    """Resets the game board and history."""
    global board, game_history_san
    board = chess.Board()
    game_history_san = []
    print("\n--- Game Reset ---\n")
    return jsonify(get_game_state().get_json())


@app.route('/api/ask_piece', methods=['POST'])
def ask_piece():
    """Handles requests to ask a specific piece for its perspective using Gemini."""
    global game_history_san # Needs history context
    if not genai_model:
        return jsonify({'error': "'Ask Piece' feature unavailable (Gemini not configured)."}), 503

    data = request.get_json()
    question = data.get('question')
    piece_code = data.get('piece')
    square = data.get('square')
    fen = data.get('fen')

    if not all([question, piece_code, square, fen]):
        return jsonify({'error': 'Missing required data (question, piece, square, fen)'}), 400
    # Basic validation of inputs
    if not (isinstance(piece_code, str) and len(piece_code)==2 and piece_code.startswith('w') and piece_code[1].upper() in 'PNBRQK'):
         return jsonify({'error': 'Invalid piece code format (e.g., wP, wN).'}), 400
    try:
        chess.parse_square(square)
        temp_board = chess.Board(fen) # Validate FEN syntax
        # Optional: Check if the piece actually matches the FEN - useful for debugging client/server mismatch
        piece_at_sq = temp_board.piece_at(chess.parse_square(square))
        expected_symbol = piece_code[1].upper() # Asking perspective of white pieces only
        if not piece_at_sq or piece_at_sq.symbol() != expected_symbol:
            print(f"Warning: Mismatch - FEN shows {piece_at_sq} at {square}, but user asked about {piece_code}")
            # Proceed anyway, but log the warning
    except ValueError as e_val:
        return jsonify({'error': f'Invalid square or FEN provided: {e_val}'}), 400
    except Exception as e_other: # Catch potential chess.Board init errors
        return jsonify({'error': f'Error validating input FEN/square: {e_other}'}), 400

    try:
        answer = get_llm_piece_perspective(fen, game_history_san, piece_code, square, question)
        return jsonify({'answer': answer})
    except Exception as e:
        print(f"Error getting piece perspective: {e}")
        traceback.print_exc()
        return jsonify({'error': 'Server error while asking piece'}), 500


# --- Frontend Serving Route ---
@app.route('/')
def index():
    # HTML content includes JS fixes for reset button
    html_content = """
<!DOCTYPE html>
<html lang="en">
<head>
    <meta charset="UTF-8">
    <meta name="viewport" content="width=device-width, initial-scale=1.0, maximum-scale=1.0, user-scalable=no">
    <title>ChessTalk (vs Stockfish)</title>
    <link rel="stylesheet" href="https://unpkg.com/@chrisoakman/chessboardjs@1.0.0/dist/chessboard-1.0.0.min.css" crossorigin="anonymous">
    <script src="https://code.jquery.com/jquery-3.6.0.min.js" integrity="sha256-/xUj+3OJU5yExlq6GSYGSHk7tPXikynS7ogEvDej/m4=" crossorigin="anonymous"></script>
    <script src="https://unpkg.com/@chrisoakman/chessboardjs@1.0.0/dist/chessboard-1.0.0.min.js" crossorigin="anonymous"></script>
    <style>
        body { font-family: sans-serif; display: flex; justify-content: center; padding: 10px; background-color: #f0f0f0; margin: 0; -webkit-tap-highlight-color: transparent; }
        .container { display: flex; flex-direction: column; align-items: center; background-color: #fff; padding: 15px; border-radius: 8px; box-shadow: 0 2px 10px rgba(0,0,0,0.1); width: 100%; max-width: 500px; box-sizing: border-box; }
        #myBoard { width: 90vw; max-width: 450px; margin-bottom: 15px; user-select: none; -webkit-user-select: none; touch-action: none; }
        h1 { margin-top: 0; margin-bottom: 5px; color: #333; font-size: 1.8em; text-align: center; }
        .subheaders { text-align: center; color: #555; margin-bottom: 10px; font-size: 1.1em; line-height: 1.4; }
        #instructions { font-size: 0.9em; color: #666; margin-bottom: 15px; text-align: center; background-color: #f9f9f9; padding: 8px; border-radius: 4px; border: 1px solid #eee;}
        .status-info { font-size: 1.1em; font-weight: bold; margin-bottom: 10px; padding: 8px 15px; border-radius: 4px; text-align: center; min-width: 200px; width: 90%; box-sizing: border-box; }
        .status-info.turn-white { background-color: #e0ffe0; border: 1px solid #90ee90; color: #006400; }
        .status-info.turn-black { background-color: #f0f0f0; border: 1px solid #cccccc; color: #333; }
        .status-info.thinking { background-color: #fffacd; border: 1px solid #ffd700; color: #b8860b; animation: pulse 1.5s infinite ease-in-out; }
        .status-info.game-over { background-color: #add8e6; border: 1px solid #87ceeb; color: #00008b; }
        .status-info.error { background-color: #ffe4e1; border: 1px solid #ffb6c1; color: #dc143c; }
        .controls { margin-bottom: 15px; }
        .button { padding: 10px 15px; font-size: 1em; border: none; border-radius: 4px; cursor: pointer; margin: 0 5px; transition: background-color 0.2s; }
        .button:disabled { background-color: #ccc; cursor: not-allowed; }
        .button-primary { background-color: #4CAF50; color: white; } .button-primary:hover:not(:disabled) { background-color: #45a049; }
        .button-secondary { background-color: #008CBA; color: white; } .button-secondary:hover:not(:disabled) { background-color: #007ba7; }
        .history-log { max-height: 120px; overflow-y: auto; border: 1px solid #ccc; padding: 10px; width: 90%; margin-top: 10px; background-color: #f9f9f9; border-radius: 4px; box-sizing: border-box; }
        .history-log h2 { margin: 0 0 10px 0; font-size: 1.1em; text-align: center; border-bottom: 1px solid #eee; padding-bottom: 5px; color: #555; }
        #history-list { list-style: none; padding: 0; margin: 0; font-size: 0.9em; }
        #history-list li { padding: 3px 0; border-bottom: 1px dotted #eee; display: flex; justify-content: space-between; } #history-list li:last-child { border-bottom: none; }
        #history-list .move-number { color: #888; font-weight: bold; flex-basis: 10%; text-align: right; padding-right: 5px;}
        #history-list .move-white { flex-basis: 45%; text-align: center; font-weight: 500;}
        #history-list .move-black { flex-basis: 45%; text-align: left; font-weight: normal;}
        .dialog-overlay { position: fixed; top: 0; left: 0; width: 100%; height: 100%; background-color: rgba(0, 0, 0, 0.6); display: none; justify-content: center; align-items: center; z-index: 1000; }
        .dialog-box { background-color: white; padding: 25px; border-radius: 8px; box-shadow: 0 5px 15px rgba(0,0,0,0.3); text-align: center; max-width: 400px; width: 90%; position: relative; box-sizing: border-box; }
        .dialog-box h3 { margin-top: 0; margin-bottom: 15px; font-size: 1.2em; color: #333; }
        .dialog-box textarea { width: 95%; min-height: 60px; margin-bottom: 15px; padding: 8px; border: 1px solid #ccc; border-radius: 4px; font-size: 1em; resize: vertical; box-sizing: border-box; }
        .dialog-box .dialog-buttons { display: flex; justify-content: space-around; }
        .dialog-response { margin-top: 15px; padding: 10px; background-color: #eef; border: 1px solid #ccd; border-radius: 4px; text-align: left; font-style: italic; max-height: 150px; overflow-y: auto; min-height: 40px; }
        .dialog-response.thinking { color: #888; display: flex; align-items: center; justify-content: center;}
        .dialog-response.error { color: #dc143c; font-weight: bold; }
        .dialog-close-btn { position: absolute; top: 10px; right: 10px; background: none; border: none; font-size: 1.5em; cursor: pointer; color: #888; line-height: 1; padding: 0;}
        .dialog-close-btn:hover { color: #333; }
        @keyframes pulse { 0% { opacity: 1; } 50% { opacity: 0.6; } 100% { opacity: 1; } }
        @media (max-width: 480px) { h1 { font-size: 1.6em; } .subheaders { font-size: 1em; } #myBoard { max-width: 320px; } .button { padding: 8px 12px; font-size: 0.9em; } .status-info { font-size: 1em; } }
    </style>
</head>
<body>
    <div class="container">
        <h1>ChessTalk</h1>
        <div class="subheaders">
            Play Chess against Stockfish<br>
            Talk to Your Pieces (White)
        </div>
        <div id="instructions">Loading instructions...</div>
        <div id="myBoard"></div>
        <div id="status" class="status-info">Loading game...</div>
        <div class="controls">
            <button id="reset-game" class="button button-secondary">Reset Game</button>
        </div>
         <div id="history" class="history-log">
            <h2>Game History</h2>
            <ol id="history-list"><li>No moves yet.</li></ol>
        </div>
    </div>

    <!-- Dialog Box HTML -->
     <div id="piece-dialog-overlay" class="dialog-overlay">
        <div class="dialog-box">
            <button id="dialog-close" class="dialog-close-btn" aria-label="Close dialog">×</button>
            <h3 id="dialog-title">Ask the Piece</h3>
            <textarea id="dialog-question" placeholder="What should I be worried about? What are my best moves? How did I end up here?" aria-label="Question for the piece"></textarea>
            <div id="dialog-response-area" class="dialog-response" style="display: none;" aria-live="polite"></div>
            <div class="dialog-buttons">
                <button id="dialog-submit" class="button button-primary">Ask</button>
                <button id="dialog-cancel" class="button button-secondary">Cancel</button>
            </div>
        </div>
     </div>

    <!-- *** START OF JAVASCRIPT *** -->
    <script>
        $(document).ready(function() {
            console.log("JS: Document ready.");
            // --- 1. Variables ---
            var board = null, currentFen = 'start', currentGameHistory = [];
            var isPlayerTurn = true, isGameOver = false, isProcessingMove = false;
            var $status = $('#status'), $historyList = $('#history-list'), $boardElement = $('#myBoard');
            var $dialogOverlay = $('#piece-dialog-overlay'), $dialogTitle = $('#dialog-title'), $dialogQuestion = $('#dialog-question'), $dialogResponseArea = $('#dialog-response-area'), $dialogSubmit = $('#dialog-submit'), $dialogCancel = $('#dialog-cancel'), $dialogClose = $('#dialog-close');
            var currentDialogData = null;
            const $resetButton = $('#reset-game'); // Cache the reset button

            // --- Interaction State ---
            const isMobile = /Mobi|Android|iPhone|iPad|iPod/i.test(navigator.userAgent) || ('ontouchstart' in window);
            var isRightMouseButtonDown = false;

            // Long Press (Mobile Dialog)
            var longPressTimer = null;
            var longPressDuration = 700;
            var touchstartX = 0, touchstartY = 0;
            var pressTargetSquare = null, pressTargetPiece = null;
            var longPressFired = false;

            const API_BASE_URL = "/api";
            console.log("JS: Mobile detected:", isMobile);

            // --- Update Instructions ---
            const $instructions = $('#instructions');
             if (isMobile) {
                $instructions.html("You play White. Swipe/Drag pieces to move.<br><b>Long-press</b> your pieces to talk!");
             } else {
                $instructions.html("You play White. Drag pieces to move.<br><b>Right-click</b> your pieces to talk!");
             }

            // --- 2. Helpers ---
            async function fetchApi(endpoint, method = 'GET', body = null) {
                const url = `${API_BASE_URL}${endpoint}`;
                const options = { method: method, headers: { 'Content-Type': 'application/json', 'Accept': 'application/json' }};
                if (body) { options.body = JSON.stringify(body); }
                console.log(`JS: Fetch ${method} ${url}`, body ? JSON.stringify(body).substring(0,100) : '');
                try {
                    const r = await fetch(url, options);
                    const d = await r.json();
                    if (!r.ok) { throw new Error(d.error || `HTTP ${r.status}`); }
                    return d;
                } catch (e) {
                    console.error(`API Error (${method} ${url}):`, e);
                    updateStatus(`Error: ${e.message || 'Network'}`, true);
                    throw e;
                }
            }
            function updateStatus(msg, isErr=false, isThink=false, baseCls=null) {
                if(!$status.length) return;
                $status.text(msg);
                $status.removeClass('error thinking turn-white turn-black game-over');
                if(baseCls) $status.addClass(baseCls);
                if(isErr) $status.addClass('error');
                if(isThink) $status.addClass('thinking');
            }
            function updateHistoryList(hist) {
                if(!$historyList.length) return;
                $historyList.empty();
                currentGameHistory = Array.isArray(hist) ? hist : [];
                if (currentGameHistory.length === 0) { $historyList.append('<li>No moves yet.</li>'); return; }
                for (let i=0; i<currentGameHistory.length; i+=2) {
                    const mn=Math.floor(i/2)+1;
                    const wm=currentGameHistory[i]||'';
                    const bm=currentGameHistory[i+1]||(isGameOver?'':'...');
                    $historyList.append(`<li><span class="move-number">${mn}.</span><span class="move-white">${wm}</span><span class="move-black">${bm}</span></li>`);
                }
                const logEl=$historyList.closest('.history-log')[0];
                if(logEl) logEl.scrollTop=logEl.scrollHeight;
            }
            function getPieceName(pc) {
                const map={'P':'Pawn','N':'Knight','B':'Bishop','R':'Rook','Q':'Queen','K':'King'};
                return (typeof pc === 'string' && pc.length === 2) ? map[pc[1].toUpperCase()]||'?' : '?';
            }

             // --- 3. State Update ---
            function updateGameState(data) {
                 console.log("JS: updateGameState called with data:", data);
                 if (data && typeof data.fen === 'string') {
                    const wasLoading = $status.text().startsWith("Loading");
                    currentFen = data.fen;
                    currentGameHistory = Array.isArray(data.history) ? data.history : [];
                    isGameOver = data.is_game_over || false;
                    isPlayerTurn = (data.turn === 'white') && !isGameOver;

                    let statusMsg = "Error"; let statusClass = "error";
                    if (isGameOver) {
                        statusClass = "game-over";
                        const term = data.termination ? `(${data.termination})` : '';
                        if (data.winner === 'white') statusMsg = `🎉 Game Over! You Won! ${term}`;
                        else if (data.winner === 'black') statusMsg = `🤖 Game Over! Opponent Won! ${term}`;
                        else statusMsg = `🤝 Game Over! Draw! ${term}`;
                    } else if (isPlayerTurn) {
                        statusClass = "turn-white"; statusMsg = "⚪ Your Turn (White)";
                    } else {
                        statusClass = "turn-black"; statusMsg = "⚫ Opponent is thinking...";
                        if (!wasLoading && $status.text() !== "🔄 Resetting game...") {
                            updateStatus(statusMsg, false, true, statusClass);
                        } else { updateStatus(statusMsg, false, false, statusClass); }
                    }

                    if (statusClass !== "turn-black" || wasLoading || $status.text() === "🔄 Resetting game..." || !$status.hasClass('thinking')) {
                         const finalStatusMsg = statusMsg; const finalStatusClass = statusClass;
                         const isThinkingStatus = (statusClass === "turn-black" && !isGameOver);
                         const updateFunc = () => { updateStatus(finalStatusMsg, false, isThinkingStatus, finalStatusClass); };
                         if (wasLoading) { setTimeout(updateFunc, 150); } else { updateFunc(); }
                    }

                    updateHistoryList(currentGameHistory);

                    if (board && typeof board.position === 'function') {
                        if(board.fen() !== currentFen) { board.position(currentFen, false); }
                    } else { console.error("JS: updateGameState - CRITICAL: Board object invalid!"); }

                    if(data.last_opponent_move && !isGameOver && isPlayerTurn) {
                         if ($status.hasClass('thinking')) {
                            updateStatus("⚪ Your Turn (White)", false, false, "turn-white");
                         }
                    }
                 } else {
                     console.error("JS: updateGameState received invalid data");
                     updateStatus("Error: Invalid game state from server.", true);
                     isPlayerTurn = false; isGameOver = true;
                 }
             }

            // --- 4. Event Handlers ---
            function clearLongPress() { if (longPressTimer) { clearTimeout(longPressTimer); longPressTimer = null; } pressTargetSquare = null; pressTargetPiece = null; }
            function onBoardMouseDown(event) { if (event.button === 2) { isRightMouseButtonDown = true; } else { isRightMouseButtonDown = false; } longPressFired = false; }
            $boardElement.on('contextmenu', (e) => { e.preventDefault(); return false; });
            function onDragStart (source, piece, position, orientation) { if (longPressFired || (!isMobile && isRightMouseButtonDown)) return false; clearLongPress(); isRightMouseButtonDown = false; if (isGameOver || !isPlayerTurn || piece.search(/^b/) !== -1 || isProcessingMove) return false; return true; }

            async function onDrop (source, target, piece, newPos, oldPos, orientation) {
                const wasRightClickAttempt = isRightMouseButtonDown;
                isRightMouseButtonDown = false; longPressFired = false; clearLongPress();
                if (source === target && wasRightClickAttempt && !isMobile) {
                    const isWhite = piece && piece.search(/^w/) !== -1;
                    if (isWhite && !isProcessingMove && isPlayerTurn && !isGameOver) { openPieceDialog(piece, source); return; }
                    else { return 'snapback'; }
                }
                else if (source !== target && !wasRightClickAttempt) {
                    if (isGameOver || !isPlayerTurn || isProcessingMove) return 'snapback';
                    var moveUCI = source + target;
                    if (piece === 'wP' && target.charAt(1) === '8') { moveUCI += 'q'; }
                    isProcessingMove = true; $resetButton.prop('disabled', true); // Disable here
                    updateStatus("⚪ Sending move...", false, true, "turn-white");
                    try {
                        const data = await fetchApi('/move', 'POST', { move: moveUCI });
                        updateGameState(data);
                    } catch (error) { if (board) board.position(currentFen, false); return 'snapback'; }
                    finally {
                        isProcessingMove = false;
                        $resetButton.prop('disabled', false); // <<< Always enable here
                    }
                }
                else { return 'snapback'; }
            }

            // --- 5. Dialog Logic ---
            function openPieceDialog(piece, square) { clearLongPress(); longPressFired = true; if (!piece || !square) return; if ($dialogOverlay.is(':visible') && currentDialogData?.piece === piece && currentDialogData?.square === square) return; currentDialogData = { piece, square }; $dialogTitle.text(`Ask the ${getPieceName(piece)} on ${square}`); $dialogQuestion.val(''); $dialogResponseArea.hide().empty().removeClass('error thinking'); $dialogSubmit.prop('disabled', false); $dialogCancel.prop('disabled', false); $dialogOverlay.css('display', 'flex'); $dialogQuestion.focus(); }
            function closePieceDialog() { $dialogOverlay.hide(); currentDialogData = null; longPressFired = false; }
            async function submitPieceQuestion() { if (!currentDialogData) return; const question = $dialogQuestion.val().trim(); if (!question) { $dialogResponseArea.text('Please enter a question.').addClass('error').show(); setTimeout(() => { if (!$dialogResponseArea.hasClass('thinking')) $dialogResponseArea.hide().removeClass('error'); }, 2500); return; } const { piece, square } = currentDialogData; $dialogResponseArea.html('<i>Thinking...</i>').addClass('thinking').removeClass('error').show(); $dialogSubmit.prop('disabled', true); $dialogCancel.prop('disabled', true); try { const data = await fetchApi('/ask_piece', 'POST', { question, piece, square, fen: currentFen }); $dialogResponseArea.text(data.answer || 'The piece offered no answer.').removeClass('thinking error'); } catch (error) { $dialogResponseArea.text(`Error: ${error.message || 'Could not get answer.'}`).addClass('error').removeClass('thinking'); } finally { $dialogSubmit.prop('disabled', false); $dialogCancel.prop('disabled', false); } }

            // --- 6. Init ---
            async function initializeGame() {
                 updateStatus("Loading game...", false, true);
                 isProcessingMove = true;
                 $resetButton.prop('disabled', true); // Disable here
                 try {
                    const data = await fetchApi('/game');
                    if (!data || typeof data.fen !== 'string' || !Array.isArray(data.history)) { throw new Error("Invalid initial game data received."); }
                    currentFen = data.fen; currentGameHistory = data.history;
                    var cfg = { draggable: true, position: currentFen, orientation: 'white', onDragStart: onDragStart, onDrop: onDrop, pieceTheme: '/static/{piece}.png', moveSpeed: 200, snapbackSpeed: 400, snapSpeed: 50, };
                    if (!$('#myBoard').length) throw new Error("Board element #myBoard not found.");
                    board = Chessboard('myBoard', cfg);
                    if (!board || typeof board.position !== 'function') { throw new Error("Chessboard.js initialization failed."); }
                    if (!isMobile) { $boardElement.on('mousedown', onBoardMouseDown); }
                    $boardElement.off('contextmenu').on('contextmenu', (e) => { e.preventDefault(); return false; });
                    if (isMobile) { $boardElement.on('touchstart', '.square-55d63', function(e) { if (e.originalEvent.touches.length > 1 || isProcessingMove || isGameOver || !isPlayerTurn) return; clearLongPress(); longPressFired = false; const s = $(this).data('square'); const p = board.position()[s] || null; pressTargetSquare = s; pressTargetPiece = p; if (p && p.startsWith('w')) { touchstartX = e.originalEvent.touches[0].pageX; touchstartY = e.originalEvent.touches[0].pageY; console.log(`JS: touchstart ${s}(${p}). Start timer.`); longPressTimer = setTimeout(() => { if (pressTargetSquare && pressTargetPiece && pressTargetPiece.startsWith('w') && !isProcessingMove && isPlayerTurn && !isGameOver) { console.log(`JS: Long press fired.`); openPieceDialog(p, s); } else { console.log("JS: Long press timeout, conditions unmet."); } longPressTimer = null; pressTargetSquare = null; pressTargetPiece = null; }, longPressDuration); } }); $boardElement.on('touchend touchcancel', '.square-55d63', function(e) { if (longPressTimer) { console.log("JS: touchend/cancel - Clearing timer."); clearLongPress(); } longPressFired = false; }); $boardElement.on('touchmove', function(e) { if (longPressTimer && e.originalEvent.touches.length === 1) { const t = 10; let cX = e.originalEvent.touches[0].pageX; let cY = e.originalEvent.touches[0].pageY; if (Math.abs(cX - touchstartX) > t || Math.abs(cY - touchstartY) > t) { console.log("JS: touchmove - Clearing timer."); clearLongPress(); } } }); }
                    $(window).resize(() => { if(board) board.resize(); }).trigger('resize');
                    updateGameState(data);
                 } catch (error) { console.error("Initialization failed:", error); updateStatus(`Initialization Failed: ${error.message || 'Unknown error'}`, true); isGameOver = true; }
                 finally {
                     isProcessingMove = false;
                     $resetButton.prop('disabled', false); // <<< Always enable here
                 }
             }

            // --- 7. Listeners ---
            $resetButton.on('click', async function() { // Use cached selector
                if (isProcessingMove) return;
                clearLongPress(); longPressFired=false;
                isProcessingMove = true; $(this).prop('disabled', true); // Disable here
                closePieceDialog();
                updateStatus("🔄 Resetting game...", false, true);
                try {
                    const data = await fetchApi('/reset', 'POST');
                    if (board) board.position('start', false);
                    updateGameState(data);
                } catch (error) { updateStatus(`Error resetting game: ${error.message || 'Unknown'}`, true); }
                finally {
                    isProcessingMove = false;
                    $(this).prop('disabled', false); // <<< Always enable here
                }
            });
            $dialogSubmit.on('click', submitPieceQuestion);
            $dialogCancel.on('click', closePieceDialog);
            $dialogClose.on('click', closePieceDialog);
            $dialogQuestion.on('keypress', (e) => { if (e.key === 'Enter' && !e.shiftKey) { e.preventDefault(); submitPieceQuestion(); } });
            $(document).on('keydown', (e) => { if (e.key === "Escape" && $dialogOverlay.is(':visible')) { closePieceDialog(); } });

            // --- 8. Start ---
            initializeGame();
        });
       </script>
    <!-- *** END OF JAVASCRIPT *** -->
    </body>
    </html>
    """
    return Response(html_content, mimetype='text/html')

# --- Main Execution ---
if __name__ == '__main__':
    print("\n--- Starting Flask Server ---")
    print(f"Stockfish Engine: {'Initialized' if stockfish_engine else 'DISABLED'}")
    if not stockfish_engine:
         print("  (Review initialization logs above for errors: Path, Permissions, Validation)")
    print(f"Gemini Model ('Ask Piece'): {'Initialized' if genai_model else 'DISABLED'}")
    if not genai_model:
         print("  (Check GOOGLE_API_KEY env variable)")
    print(f"Opponent Moves: {'Stockfish (if Initialized)' if stockfish_engine else 'Random (Stockfish Disabled)'}")
    if stockfish_engine:
        print(f"Stockfish Think Time: {STOCKFISH_THINK_TIME_MS}ms per move")
    print("Features: Talk to Pieces (Right-Click Desktop / Long-Press Mobile), Drag/Swipe Moves")
    print("Access at: http://127.0.0.1:5000/ (or your local network IP:5000 if using host 0.0.0.0)")
    print("-----------------------------\n")
    # use_reloader=False is recommended for stability with external processes like Stockfish
    app.run(host='0.0.0.0', port=5000, debug=True, use_reloader=False)
