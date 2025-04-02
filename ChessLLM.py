import os
import chess
import chess.pgn
import google.generativeai as genai
import random
import time
from flask import Flask, request, jsonify, Response # Make sure Flask is imported
from flask_cors import CORS
from dotenv import load_dotenv
import traceback
import requests

# --- Constants ---
LICHESS_EXPLORER_URL = "https://explorer.lichess.ovh/lichess"
OPENING_MOVE_LIMIT = 10

# --- Environment & API Key Setup ---
load_dotenv()
GEMINI_API_KEY = os.getenv("GOOGLE_API_KEY") or os.getenv("CURSOR_GOOGLE_API_KEY")
genai_model = None
# Corrected model name based on standard availability (flash is generally available)
# If you have access to 'gemini-2.0-flash', keep that, otherwise use 'gemini-1.5-flash-latest'.
MODEL_NAME = 'gemini-1.5-flash-latest'
# MODEL_NAME = 'gemini-2.0-flash' # Use if you have specific access

if GEMINI_API_KEY:
    try:
        genai.configure(api_key=GEMINI_API_KEY)
        genai_model = genai.GenerativeModel(MODEL_NAME)
        print(f"Gemini Model Initialized ({MODEL_NAME}).")
    except Exception as e:
        print(f"Error initializing Gemini model: {e}. LLM disabled.")
        # traceback.print_exc() # Optionally uncomment for more detailed init errors
else:
    print("Warning: GOOGLE_API_KEY environment variable not set. LLM disabled.")

# --- Flask App Setup ---
app = Flask(__name__, static_folder='static', static_url_path='/static')
CORS(app, resources={r"/api/*": {"origins": "*"}}) # Apply CORS after creating app

# --- Game State ---
board = chess.Board()
game_history_san = []

# --- Helper Functions ---
def format_history_as_pgn_movetext(history_san):
    # ...(keep existing code)...
    if not history_san: return "(No moves played yet)"
    pgn_string = ""
    for i, move in enumerate(history_san):
        move_number = (i // 2) + 1
        if i % 2 == 0: pgn_string += f"{move_number}. {move} "
        else: pgn_string += f"{move} "
    return pgn_string.strip()

def get_lichess_opening_move(current_board):
    # ...(keep existing code)...
    uci_moves = [move.uci() for move in current_board.move_stack]
    play_param = ",".join(uci_moves)
    params = {'variant': 'standard', 'play': play_param}
    headers = {'Accept': 'application/json'}
    print(f"Requesting Lichess opening move (History: {play_param})...")
    try:
        response = requests.get(LICHESS_EXPLORER_URL, params=params, headers=headers, timeout=5)
        response.raise_for_status()
        data = response.json()
        moves_data = data.get('moves', [])
        if not moves_data: print("Lichess Explorer returned no moves."); return None
        total_games = sum(m.get('white', 0) + m.get('draws', 0) + m.get('black', 0) for m in moves_data)
        chosen_move_info = None # Initialize
        if total_games == 0 and moves_data: chosen_move_info = moves_data[0]
        elif total_games > 0:
            weights = [(m.get('white', 0) + m.get('draws', 0) + m.get('black', 0)) for m in moves_data]
            # Ensure weights and moves_data align before random.choices
            if len(weights) == len(moves_data):
                try:
                    chosen_move_info = random.choices(moves_data, weights=weights, k=1)[0]
                except ValueError as e:
                    # Handle potential issues if weights are all zero or invalid
                    print(f"Lichess weighted choice error: {e}, picking first move.")
                    if moves_data: chosen_move_info = moves_data[0]
            else:
                print("Lichess weight/move count mismatch, picking first.")
                if moves_data: chosen_move_info = moves_data[0]
        else:
             print("Lichess: No moves data or zero total games.")
             return None

        if not chosen_move_info:
             print("Lichess: Could not determine a chosen move.")
             return None

        chosen_uci = chosen_move_info.get('uci')
        if not chosen_uci:
             print("Lichess: Chosen move info lacks UCI.")
             return None

        try:
            move = current_board.parse_uci(chosen_uci)
            if move in current_board.legal_moves:
                san_move = current_board.san(move)
                print(f"Lichess selected: {san_move}")
                return san_move
            else:
                print(f"Lichess proposed illegal move {chosen_uci}")
                return None
        except ValueError:
            print(f"Lichess gave invalid UCI {chosen_uci}")
            return None
    except requests.exceptions.RequestException as e: print(f"Lichess API Error: {e}"); return None
    except ValueError as e: print(f"Lichess JSON Decode Error: {e}"); return None # Catches JSONDecodeError
    except Exception as e: print(f"Lichess Unexpected Error: {e}"); traceback.print_exc(); return None


def get_llm_move(current_board, history_san):
    # ...(keep existing code)...
    if not genai_model:
        print("LLM N/A, random.")
        legal_moves_san = [current_board.san(m) for m in current_board.legal_moves]
        return random.choice(legal_moves_san) if legal_moves_san else None

    legal_moves_san = [current_board.san(m) for m in current_board.legal_moves]
    if not legal_moves_san: return None
    fen = current_board.fen(); pgn_movetext = format_history_as_pgn_movetext(history_san)
    prompt = f"""Expert chess analyst simulating strong GM (Black). Goal: BEST move. ANALYSIS (Internal): 1. Tactical Safety Scan: Checks, captures, threats, undefended pieces, combinations? 2. Forcing Moves: Your checks, captures, threats. Evaluate first. 3. Deep Calculation: Promising lines 3-4 ply deep. 4. Positional Eval: King safety, material, activity, center, pawns, files. 5. Strategic Context ({pgn_movetext}): Plans? Imbalances? Direction? 6. Candidate Comparison: Top 2-3 for safety/tactics/position. DO NOT: Passive moves if active/forcing exist. Blunder. Hope chess. Select without calc/justification. CONTEXT: FEN: {fen}. History: {pgn_movetext}. Turn: Black. Legal (SAN): {', '.join(legal_moves_san)}. ACTION: Single strongest move (SAN ONLY). Example: Qxb2"""
    print("Requesting LLM move...")

    try:
        attempts = 0; max_attempts = 3; llm_move_san = None
        while attempts < max_attempts:
            attempts += 1; print(f"LLM Attempt {attempts}/{max_attempts}")
            gen_config = genai.types.GenerationConfig(max_output_tokens=10, temperature=0.35)
            if not genai_model: raise Exception("Gemini model not initialized") # Check again inside loop
            response = genai_model.generate_content(prompt, generation_config=gen_config)
            candidate_san = response.text.strip(); print(f"LLM Raw: '{candidate_san}'")

            if not candidate_san:
                print("LLM returned empty response."); time.sleep(0.5); continue

            potential_move = None
            # Improved SAN extraction: prioritize full matches
            for part in candidate_san.split():
                cleaned_part = part.rstrip('.,;!?"\'')
                if chess.SAN_REGEX.match(cleaned_part):
                    # Check if it parses immediately to avoid ambiguity later if possible
                    try:
                         move_test = current_board.parse_san(cleaned_part)
                         if move_test in current_board.legal_moves:
                              potential_move = cleaned_part
                              print(f"Potential SAN found and validated: '{potential_move}' from part '{part}'")
                              break # Found a good one
                    except ValueError:
                         # If it looks like SAN but doesn't parse, store it as a backup
                         if potential_move is None:
                              potential_move = cleaned_part
                              print(f"Potential SAN found (validation deferred): '{potential_move}' from part '{part}'")
                         continue # Keep checking other parts for a better match

            if not potential_move:
                print(f"LLM response '{candidate_san}' doesn't contain recognizable SAN."); time.sleep(0.5); continue

            try:
                move = current_board.parse_san(potential_move)
                if move in current_board.legal_moves:
                    llm_move_san = potential_move
                    print(f"LLM proposed valid move: {llm_move_san}"); break
                else:
                    print(f"LLM proposed illegal move: {potential_move}")
            except ValueError as e_parse:
                 print(f"LLM proposed invalid/ambiguous SAN: {potential_move} ({e_parse})")
            except Exception as e_other:
                 print(f"Unexpected error parsing LLM SAN '{potential_move}': {e_other}")

            time.sleep(0.5) # Delay before retry only if validation failed

        if llm_move_san: return llm_move_san
        else:
            print("LLM failed to provide valid move after multiple attempts, choosing random.")
            legal_moves = [current_board.san(m) for m in current_board.legal_moves] # Get fresh list
            return random.choice(legal_moves) if legal_moves else None

    except Exception as e:
        print(f"Error calling Gemini API for move: {e}"); traceback.print_exc()
        print("Choosing random move due to API error.")
        legal_moves = [current_board.san(m) for m in current_board.legal_moves]
        return random.choice(legal_moves) if legal_moves else None


def get_llm_piece_perspective(fen, history_san, piece_code, square, question):
    # ...(keep existing code)...
    if not genai_model: return "LLM unavailable."
    piece_map = {'P':'Pawn','N':'Knight','B':'Bishop','R':'Rook','Q':'Queen','K':'King'}
    piece_type = piece_map.get(piece_code[1].upper(), 'Unknown Piece')
    pgn_movetext = format_history_as_pgn_movetext(history_san)
    prompt = f"""You ARE the White {piece_type} currently located on square {square}.
The current board state (FEN) is: {fen}.
The game history (PGN) is: {pgn_movetext}.

Answer the user's question strictly from your own perspective as this specific piece on {square}.
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

    print(f"Requesting LLM piece perspective ({piece_type}@{square}) with CONSTRAINED prompt...")
    try:
        gen_config = genai.types.GenerationConfig(max_output_tokens=150, temperature=0.6)
        if not genai_model: raise Exception("Gemini model not initialized")
        response = genai_model.generate_content(prompt, generation_config=gen_config)
        answer = response.text.strip()
        prefixes_to_remove = [
            f"As the {piece_type} on {square},", f"Okay, speaking as the {piece_type} on {square}:",
            "Okay, here's my perspective:", "Alright, here's what I see:",
            f"I am the {piece_type} on {square}.", "My response (as the",
        ]
        original_answer = answer
        for prefix in prefixes_to_remove:
            if answer.lower().startswith(prefix.lower()):
                answer = answer[len(prefix):].lstrip(" ,:.-"); break
        print(f"LLM Raw Piece Response: '{original_answer}'")
        print(f"LLM Cleaned Piece Response: '{answer}'")
        return answer if answer else "I have no thoughts on that right now."
    except Exception as e:
        print(f"Gemini piece API Error: {e}"); traceback.print_exc()
        return "My thoughts are clouded... (API Error)"

# --- API Endpoints ---
@app.route('/api/game', methods=['GET'])
def get_game_state():
    # ...(keep existing code)...
    global board, game_history_san
    winner, termination = None, None; is_game_over = board.is_game_over(claim_draw=True)
    if is_game_over:
        outcome = board.outcome(claim_draw=True);
        if outcome: winner = "white" if outcome.winner == chess.WHITE else "black" if outcome.winner == chess.BLACK else "draw"; termination = outcome.termination.name.capitalize().replace('_', ' ')
    if not game_history_san and board.move_stack:
       print("Warning: Game history empty but move stack exists. Resetting board state.")
       board = chess.Board()
    return jsonify({'fen': board.fen(), 'turn': 'white' if board.turn == chess.WHITE else 'black', 'is_game_over': is_game_over, 'winner': winner, 'termination': termination, 'history': game_history_san})


@app.route('/api/move', methods=['POST'])
def make_move():
    # ...(keep existing code)...
    global board, game_history_san
    if board.turn != chess.WHITE: return jsonify({'error': 'Not White\'s turn'}), 400
    if board.is_game_over(claim_draw=True): return jsonify({'error': 'Game over'}), 400
    data = request.get_json(); uci_move_str = data.get('move')
    if not uci_move_str or not (4 <= len(uci_move_str) <= 5): return jsonify({'error': f'Invalid UCI: {uci_move_str}'}), 400
    opponent_san_response, move_source = None, None
    try:
        move = board.parse_uci(uci_move_str)
        if move in board.legal_moves:
            san_move = board.san(move); board.push(move); game_history_san.append(san_move)
            print(f"User (W): {uci_move_str} ({san_move}) | Ply: {len(game_history_san)}")
            if not board.is_game_over(claim_draw=True):
                if len(game_history_san) < OPENING_MOVE_LIMIT:
                    opponent_san = get_lichess_opening_move(board)
                    if opponent_san: opponent_san_response, move_source = opponent_san, "Lichess"
                    else: print("Lichess fail->Gemini"); opponent_san_response, move_source = get_llm_move(board, game_history_san), "Gemini(Fallback)"
                else: opponent_san_response, move_source = get_llm_move(board, game_history_san), "Gemini"

                if not opponent_san_response:
                    print(f"{move_source or 'MoveGen'} failed or returned nothing. Attempting random.")
                    legal_moves = [board.san(m) for m in board.legal_moves];
                    if legal_moves: opponent_san_response, move_source = random.choice(legal_moves), "Random(Fallback)"
                    else: print("Error: No legal moves for opponent?"); opponent_san_response = None

                if opponent_san_response:
                    try:
                        opp_move = board.parse_san(opponent_san_response);
                        if opp_move in board.legal_moves: board.push(opp_move); game_history_san.append(opponent_san_response); print(f"Opponent ({move_source}): {opponent_san_response}")
                        else: print(f"ERROR: {move_source} proposed illegal move {opponent_san_response}. Board state: {board.fen()}")
                    except ValueError as e: print(f"ERROR: Parsing {move_source} SAN '{opponent_san_response}': {e}")
                    except Exception as e: print(f"ERROR: Pushing {move_source} move '{opponent_san_response}': {e}")
                else: print(f"Opponent ({move_source or 'System'}) could not generate a move.")

            final_state_data = get_game_state().get_json();
            final_state_data['last_llm_move'] = opponent_san_response
            return jsonify(final_state_data)
        else:
            try: illegal_san = board.san(move)
            except: illegal_san = uci_move_str;
            return jsonify({'error': f'Illegal move: {illegal_san}'}), 400
    except ValueError: return jsonify({'error': f'Invalid move notation: {uci_move_str}'}), 400
    except Exception as e: print(f"Error during move processing: {e}"); traceback.print_exc(); return jsonify({'error': 'Server error during move processing'}), 500


@app.route('/api/reset', methods=['POST'])
def reset_game():
    # ...(keep existing code)...
    global board, game_history_san; board = chess.Board(); game_history_san = []; print("Game Reset."); return jsonify(get_game_state().get_json())


@app.route('/api/ask_piece', methods=['POST'])
def ask_piece():
    # ...(keep existing code)...
    global game_history_san;
    if not genai_model: return jsonify({'error': 'LLM is not available.'}), 503
    data = request.get_json();
    question, piece_code, square, fen = data.get('question'), data.get('piece'), data.get('square'), data.get('fen')
    if not all([question, piece_code, square, fen]): return jsonify({'error': 'Missing required data (question, piece, square, fen)'}), 400
    if not (piece_code.startswith('w') and len(piece_code)==2 and piece_code[1] in 'PNBRQK'): return jsonify({'error': 'Invalid piece code format.'}), 400
    try:
        chess.parse_square(square)
        temp_board = chess.Board(fen)
        piece_at_sq = temp_board.piece_at(chess.parse_square(square))
        expected_symbol = piece_code[1].upper() if piece_code.startswith('w') else piece_code[1].lower()
        if not piece_at_sq or piece_at_sq.symbol() != expected_symbol:
            print(f"Warning: Piece mismatch. FEN shows {piece_at_sq} at {square}, user asked about {piece_code}")
    except ValueError as e: return jsonify({'error': f'Invalid square or FEN provided: {e}'}), 400
    except Exception as e: return jsonify({'error': f'Error validating input: {e}'}), 400

    try:
        answer = get_llm_piece_perspective(fen, game_history_san, piece_code, square, question)
        return jsonify({'answer': answer})
    except Exception as e:
        print(f"Error getting piece perspective: {e}"); traceback.print_exc();
        return jsonify({'error': 'Server error while asking piece'}), 500


# --- Frontend Serving Route ---
@app.route('/')
def index():
    # Correctly indented Python block starts here
    html_content = """
<!DOCTYPE html>
<html lang="en">
<head>
    <meta charset="UTF-8">
    <meta name="viewport" content="width=device-width, initial-scale=1.0, maximum-scale=1.0, user-scalable=no">
    <title>ChessTalk</title>
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
            Play Chess against Gemini<br>
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

    <!-- *** START OF JAVASCRIPT (inside the Python string) *** -->
    <script>
        $(document).ready(function() {
            console.log("JS: Document ready.");
            // --- 1. Variables ---
            var board = null, currentFen = 'start', currentGameHistory = [];
            var isPlayerTurn = true, isGameOver = false, isProcessingMove = false;
            var $status = $('#status'), $historyList = $('#history-list'), $boardElement = $('#myBoard');
            var $dialogOverlay = $('#piece-dialog-overlay'), $dialogTitle = $('#dialog-title'), $dialogQuestion = $('#dialog-question'), $dialogResponseArea = $('#dialog-response-area'), $dialogSubmit = $('#dialog-submit'), $dialogCancel = $('#dialog-cancel'), $dialogClose = $('#dialog-close');
            var currentDialogData = null;

            // --- Interaction State ---
            const isMobile = /Mobi|Android|iPhone|iPad|iPod/i.test(navigator.userAgent) || ('ontouchstart' in window);
            var isRightMouseButtonDown = false; // Desktop right-click flag

            // Long Press (Mobile Dialog)
            var longPressTimer = null;
            var longPressDuration = 700; // milliseconds
            var touchstartX = 0, touchstartY = 0;
            var pressTargetSquare = null, pressTargetPiece = null;
            var longPressFired = false; // Flag to prevent drag immediately after long press dialog

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
                 console.log("JS: updateGameState called");
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

                    if (statusClass !== "turn-black" || wasLoading || $status.text() === "🔄 Resetting game...") {
                         const finalStatusMsg = statusMsg; const finalStatusClass = statusClass;
                         const isThinkingStatus = false;
                         const updateFunc = () => { updateStatus(finalStatusMsg, false, isThinkingStatus, finalStatusClass); };
                         if (wasLoading) { setTimeout(updateFunc, 150); } else { updateFunc(); }
                    }

                    updateHistoryList(currentGameHistory);

                    if (board && typeof board.position === 'function') {
                        if(board.fen() !== currentFen) { board.position(currentFen, false); }
                    } else { console.error("JS: updateGameState - CRITICAL: Board object invalid!"); }

                    if(data.last_llm_move && !isGameOver && statusClass !== "turn-white") {
                         updateStatus("⚪ Your Turn (White)", false, false, "turn-white");
                    }
                 } else {
                     console.error("JS: updateGameState received invalid data");
                     updateStatus("Error: Invalid game state from server.", true);
                     isPlayerTurn = false; isGameOver = true;
                 }
             }

            // --- 4. Event Handlers ---
            function clearLongPress() {
                 if (longPressTimer) { clearTimeout(longPressTimer); longPressTimer = null; }
                 pressTargetSquare = null; pressTargetPiece = null;
            }

            function onBoardMouseDown(event) { // Desktop only
                if (event.button === 2) { isRightMouseButtonDown = true; }
                else { isRightMouseButtonDown = false; }
                 longPressFired = false; // Reset flag
            }

            $boardElement.on('contextmenu', (e) => { e.preventDefault(); return false; });

            function onDragStart (source, piece, position, orientation) {
                console.log(`JS: onDragStart - Src:${source}, P:${piece}, Mobile:${isMobile}, LongPress:${longPressFired}, RClick:${isRightMouseButtonDown}`);
                if (longPressFired) { console.log("JS: Drag prevented post-longpress."); return false; }
                if (!isMobile && isRightMouseButtonDown) { console.log("JS: Drag prevented RClick."); return false; }
                clearLongPress(); isRightMouseButtonDown = false;
                if (isGameOver || !isPlayerTurn || piece.search(/^b/) !== -1 || isProcessingMove) { return false; }
                console.log(`JS: Drag allowed for ${piece} from ${source}`);
                return true;
            }

            async function onDrop (source, target, piece, newPos, oldPos, orientation) {
                console.log(`JS: onDrop - Src:${source}, Tgt:${target}, P:${piece}, WasRClick:${isRightMouseButtonDown}, Mobile:${isMobile}`);
                const wasRightClickAttempt = isRightMouseButtonDown;
                isRightMouseButtonDown = false; longPressFired = false; clearLongPress();

                if (source === target && wasRightClickAttempt && !isMobile) {
                    console.log("JS: Right-click release on same square.");
                    const isWhite = piece && piece.search(/^w/) !== -1;
                    if (isWhite && !isProcessingMove && isPlayerTurn && !isGameOver) {
                        openPieceDialog(piece, source); return;
                    } else { return 'snapback'; }
                }
                else if (source !== target && !wasRightClickAttempt) {
                    console.log("JS: Processing move.");
                    if (isGameOver || !isPlayerTurn || isProcessingMove) return 'snapback';
                    var moveUCI = source + target;
                    if (piece === 'wP' && target.charAt(1) === '8') { moveUCI += 'q'; }
                    isProcessingMove = true; $('#reset-game').prop('disabled', true);
                    updateStatus("⚪ Sending move...", false, true, "turn-white");
                    try {
                        const data = await fetchApi('/move', 'POST', { move: moveUCI });
                        updateGameState(data);
                    } catch (error) {
                        if (board) board.position(currentFen, false); return 'snapback';
                    } finally {
                        isProcessingMove = false; $('#reset-game').prop('disabled', isGameOver);
                    }
                }
                else { console.log("JS: Snapback condition."); return 'snapback'; }
            }

            // --- 5. Dialog Logic ---
            function openPieceDialog(piece, square) {
                 clearLongPress(); longPressFired = true; // Set flag when dialog opens via long press
                 if (!piece || !square) return;
                 if ($dialogOverlay.is(':visible') && currentDialogData?.piece === piece && currentDialogData?.square === square) return;
                 console.log(`JS: Opening dialog for ${piece} on ${square}`);
                 currentDialogData = { piece, square }; $dialogTitle.text(`Ask the ${getPieceName(piece)} on ${square}`);
                 $dialogQuestion.val(''); $dialogResponseArea.hide().empty().removeClass('error thinking');
                 $dialogSubmit.prop('disabled', false); $dialogCancel.prop('disabled', false);
                 $dialogOverlay.css('display', 'flex'); $dialogQuestion.focus();
            }
            function closePieceDialog() { $dialogOverlay.hide(); currentDialogData = null; longPressFired = false; /* Reset flag on close */ } // Reset flag
            async function submitPieceQuestion() { /* ... keep existing ... */ if (!currentDialogData) return; const question = $dialogQuestion.val().trim(); if (!question) { $dialogResponseArea.text('Please enter a question.').addClass('error').show(); setTimeout(() => { if (!$dialogResponseArea.hasClass('thinking')) $dialogResponseArea.hide().removeClass('error'); }, 2500); return; } const { piece, square } = currentDialogData; $dialogResponseArea.html('<i>Thinking...</i>').addClass('thinking').removeClass('error').show(); $dialogSubmit.prop('disabled', true); $dialogCancel.prop('disabled', true); try { const data = await fetchApi('/ask_piece', 'POST', { question, piece, square, fen: currentFen }); $dialogResponseArea.text(data.answer || 'The piece offered no answer.').removeClass('thinking error'); } catch (error) { $dialogResponseArea.text(`Error: ${error.message || 'Could not get answer.'}`).addClass('error').removeClass('thinking'); } finally { $dialogSubmit.prop('disabled', false); $dialogCancel.prop('disabled', false); } }

            // --- 6. Init ---
             async function initializeGame() {
                 updateStatus("Loading game...", false, true); isProcessingMove = true; $('#reset-game').prop('disabled', true);
                 try {
                    const data = await fetchApi('/game');
                    if (!data||!data.fen||!data.history) throw new Error("Invalid initial game data received.");
                    currentFen = data.fen; currentGameHistory = data.history;

                    var cfg = { draggable: true, position: currentFen, orientation: 'white', onDragStart: onDragStart, onDrop: onDrop, pieceTheme: '/static/{piece}.png', moveSpeed: 200, snapbackSpeed: 400, snapSpeed: 50 };
                    if (!$('#myBoard').length) throw new Error("Board element #myBoard not found.");
                    board = Chessboard('myBoard', cfg);
                    if (!board || typeof board.position !== 'function') throw new Error("Chessboard initialization failed.");

                    // --- Attach Platform-Specific Listeners ---
                    if (!isMobile) { $boardElement.on('mousedown', onBoardMouseDown); } // Desktop RClick detect
                    $boardElement.off('contextmenu').on('contextmenu', (e) => { e.preventDefault(); return false; }); // All platforms context prevent

                    if (isMobile) { // Mobile Long Press detect
                        $boardElement.on('touchstart', '.square-55d63', function(e) {
                             if (e.originalEvent.touches.length > 1 || isProcessingMove || isGameOver || !isPlayerTurn) return;
                             clearLongPress(); longPressFired = false;
                             const squareId = $(this).data('square'), pieceCode = board.position()[squareId] || null;
                             pressTargetSquare = squareId; pressTargetPiece = pieceCode;
                             if (pieceCode && pieceCode.startsWith('w')) {
                                touchstartX = e.originalEvent.touches[0].pageX; touchstartY = e.originalEvent.touches[0].pageY;
                                console.log(`JS: touchstart ${squareId}(${pieceCode}). Start timer.`);
                                longPressTimer = setTimeout(() => {
                                    if (pressTargetSquare && pressTargetPiece && pressTargetPiece.startsWith('w') && !isProcessingMove && isPlayerTurn && !isGameOver) {
                                         openPieceDialog(pressTargetPiece, pressTargetSquare); // Sets longPressFired=true
                                    }
                                    longPressTimer = null; pressTargetSquare = null; pressTargetPiece = null;
                                }, longPressDuration);
                             }
                        });
                        $boardElement.on('touchend touchcancel', '.square-55d63', function(e) {
                             clearLongPress(); longPressFired = false; // Reset flag fully on touchend/cancel
                        });
                         $boardElement.on('touchmove', function(e) {
                             if (longPressTimer && e.originalEvent.touches.length === 1) {
                                  const threshold = 10;
                                  let currentX = e.originalEvent.touches[0].pageX, currentY = e.originalEvent.touches[0].pageY;
                                  if (Math.abs(currentX - touchstartX) > threshold || Math.abs(currentY - touchstartY) > threshold) { clearLongPress(); }
                             }
                        });
                    }
                    $(window).resize(() => { if(board) board.resize(); }).trigger('resize');
                    updateGameState(data);
                 } catch (error) {
                     console.error("Initialization failed:", error); updateStatus(`Initialization Failed: ${error.message || 'Unknown error'}`, true); isGameOver = true;
                 } finally { isProcessingMove = false; $('#reset-game').prop('disabled', isGameOver); }
             }

            // --- 7. Listeners ---
            $('#reset-game').on('click', async function() { if (isProcessingMove) return; clearLongPress(); longPressFired=false; isProcessingMove = true; $(this).prop('disabled', true); closePieceDialog(); updateStatus("🔄 Resetting game...", false, true); try { const data = await fetchApi('/reset', 'POST'); if (board) board.position('start', false); updateGameState(data); } catch (error) {} finally { isProcessingMove = false; $(this).prop('disabled', false); } });
            $dialogSubmit.on('click', submitPieceQuestion); $dialogCancel.on('click', closePieceDialog); $dialogClose.on('click', closePieceDialog);
            $dialogQuestion.on('keypress', (e) => { if (e.key==='Enter' && !e.shiftKey) { e.preventDefault(); submitPieceQuestion(); }});
            $(document).on('keydown', (e) => { if (e.key === "Escape" && $dialogOverlay.is(':visible')) { closePieceDialog(); }});

            // --- 8. Start ---
            initializeGame();
        });
       </script>
    <!-- *** END OF JAVASCRIPT *** -->
    </body>
    </html>
    """
    # Correctly indented Python return statement
    return Response(html_content, mimetype='text/html')

# --- Main Execution ---
# Correctly placed and single __main__ block
if __name__ == '__main__':
    print("--- Starting Flask Server ---")
    print(f"API Key Loaded: {'Yes' if GEMINI_API_KEY else 'No'}")
    print(f"Gemini Model: {MODEL_NAME if genai_model else 'Not Initialized'}")
    print(f"Opponent Moves: Lichess API for first {OPENING_MOVE_LIMIT} plies, then {MODEL_NAME if genai_model else 'Random'}")
    print("Features: Talk to Pieces (Right-Click Desktop / Long-Press Mobile), Drag/Swipe Moves")
    print("Access at: http://127.0.0.1:5000/ (or your local network IP:5000 if using host 0.0.0.0)")
    print("-----------------------------")
    app.run(host='0.0.0.0', port=5000, debug=True, use_reloader=True)
