import os
import chess
import chess.pgn
import google.generativeai as genai
import random
import time
from flask import Flask, request, jsonify, Response
from flask_cors import CORS
from dotenv import load_dotenv
import traceback
import requests # <-- Import requests library

# --- Constants ---
LICHESS_EXPLORER_URL = "https://explorer.lichess.ovh/lichess"
OPENING_MOVE_LIMIT = 10 # Use Lichess for first 10 plies (5 full moves)

# --- Environment & API Key Setup ---
load_dotenv()
GEMINI_API_KEY = os.getenv("GOOGLE_API_KEY") or os.getenv("CURSOR_GOOGLE_API_KEY")
genai_model = None
MODEL_NAME = 'gemini-1.5-flash-latest'

if GEMINI_API_KEY:
    try:
        genai.configure(api_key=GEMINI_API_KEY)
        genai_model = genai.GenerativeModel(MODEL_NAME)
        print(f"Gemini Model Initialized ({MODEL_NAME}).")
    except Exception as e:
        print(f"Error initializing Gemini model: {e}. LLM disabled.")
        traceback.print_exc()
else:
    print("Warning: GOOGLE_API_KEY environment variable not set. LLM disabled.")

# --- Flask App Setup ---
app = Flask(__name__, static_folder='static', static_url_path='/static')
CORS(app, resources={r"/api/*": {"origins": "*"}})

# --- Game State ---
board = chess.Board()
game_history_san = []

# --- Helper to format history as PGN movetext (No change) ---
def format_history_as_pgn_movetext(history_san):
    if not history_san: return "(No moves played yet)"
    pgn_string = ""
    for i, move in enumerate(history_san):
        move_number = (i // 2) + 1
        if i % 2 == 0: pgn_string += f"{move_number}. {move} "
        else: pgn_string += f"{move} "
    return pgn_string.strip()

# --- NEW: Lichess Opening Explorer Helper ---
def get_lichess_opening_move(current_board):
    """
    Fetches a popular opening move from the Lichess explorer for the current position.
    Returns the move in SAN format, or None if failed or no moves found.
    """
    # Get move history in UCI format for the API query parameter
    uci_moves = [move.uci() for move in current_board.move_stack]
    play_param = ",".join(uci_moves)

    # FEN can also be used, but 'play' is often good for openings
    # fen_param = current_board.fen()
    # params = {'variant': 'standard', 'fen': fen_param}

    params = {
        'variant': 'standard',
        'play': play_param
    }
    headers = {
        'Accept': 'application/json'
    }

    print(f"Requesting Lichess opening move (History: {play_param})...")
    try:
        response = requests.get(LICHESS_EXPLORER_URL, params=params, headers=headers, timeout=5) # 5 second timeout
        response.raise_for_status() # Raise HTTPError for bad responses (4xx or 5xx)
        data = response.json()

        moves_data = data.get('moves', [])
        if not moves_data:
            print("Lichess Explorer returned no moves for this position.")
            return None

        # --- Weighted Random Selection ---
        # Calculate total games for weighting
        total_games = sum(m.get('white', 0) + m.get('draws', 0) + m.get('black', 0) for m in moves_data)
        if total_games == 0:
             # If no games recorded (unlikely for early moves), pick the first listed move
             print("Lichess Explorer: No game counts found, picking first listed move.")
             chosen_move_info = moves_data[0]
        else:
            weights = [(m.get('white', 0) + m.get('draws', 0) + m.get('black', 0)) / total_games for m in moves_data]
            # Filter out moves with zero weight if necessary, though random.choices handles it
            valid_moves_indices = [i for i, w in enumerate(weights) if w > 0]
            if not valid_moves_indices:
                 print("Lichess Explorer: No moves with positive game counts, picking first.")
                 chosen_move_info = moves_data[0]
            else:
                 filtered_moves = [moves_data[i] for i in valid_moves_indices]
                 filtered_weights = [weights[i] for i in valid_moves_indices]
                 # Normalize weights just in case (shouldn't be needed but safe)
                 sum_filtered_weights = sum(filtered_weights)
                 normalized_weights = [w / sum_filtered_weights for w in filtered_weights]

                 chosen_move_info = random.choices(filtered_moves, weights=normalized_weights, k=1)[0]

        chosen_uci = chosen_move_info.get('uci')
        if not chosen_uci:
             print("Lichess Explorer: Chosen move data missing UCI.")
             return None

        # Validate and convert UCI to SAN using the current board state
        try:
            move = current_board.parse_uci(chosen_uci)
            if move in current_board.legal_moves:
                san_move = current_board.san(move)
                print(f"Lichess selected move: {san_move} ({chosen_uci})")
                return san_move
            else:
                # This shouldn't happen if Lichess data is for the right position
                print(f"Lichess proposed an illegal move: {chosen_uci}?")
                return None
        except ValueError:
            print(f"Lichess proposed invalid UCI: {chosen_uci}")
            return None

    except requests.exceptions.RequestException as e:
        print(f"Error calling Lichess API: {e}")
        return None
    except ValueError as e: # Catches JSONDecodeError
        print(f"Error decoding Lichess API response: {e}")
        return None
    except Exception as e:
        print(f"Unexpected error getting Lichess move: {e}")
        traceback.print_exc()
        return None

# --- LLM Helper for Black Moves (No change from previous version) ---
def get_llm_move(current_board, history_san):
    # (Keep the strong prompt from the previous step)
    if not genai_model:
        print("LLM not available. Choosing random move.")
        legal_moves_san = [current_board.san(move) for move in current_board.legal_moves]
        return random.choice(legal_moves_san) if legal_moves_san else None

    legal_moves_san = [current_board.san(move) for move in current_board.legal_moves]
    if not legal_moves_san: return None

    fen = current_board.fen()
    pgn_movetext = format_history_as_pgn_movetext(history_san)

    # --- REVISED, MORE DEMANDING PROMPT (FROM PREVIOUS STEP) ---
    prompt = f"""You are an expert chess analyst simulating a strong Grandmaster playing as Black. Your goal is uncompromising: find and play the single BEST move. Mediocrity is unacceptable.

CRITICAL ANALYSIS REQUIRED (Internal Thought Process - Do not output this part):
1.  **Tactical Safety Scan:** IMMEDIATELY check for checks, captures, and major threats against your position. Identify ALL undefended pieces (yours and opponent's). Are there any pending sacrifices or combinations for either side?
2.  **Forcing Moves:** Identify ALL your possible checks, captures, and direct threats. Evaluate these first. Often the best move is tactical.
3.  **Deep Calculation:** For promising tactical lines and key positional moves, calculate variations at least 3-4 ply deep (your move, opponent's reply, your response, opponent's reply).
4.  **Positional Evaluation:** Assess King safety (yours AND opponent's), material balance, piece activity/coordination, central control/space, pawn structure strengths/weaknesses, open files/diagonals.
5.  **Strategic Context (History):** Review the game history ({pgn_movetext}). What were the likely plans? How did the current imbalances arise? Does the history suggest a specific strategic direction (e.g., attack, defense, simplification)?
6.  **Candidate Move Comparison:** Compare the top 2-3 candidate moves resulting from your analysis. Which one offers the best combination of safety, tactical advantage, and long-term positional gain?

DO NOT:
*   Make passive moves if active, forcing, or initiative-gaining moves exist.
*   Blunder material or overlook simple tactics (forks, pins, skewers).
*   Play hope chess. Assume the opponent will find the best reply.
*   Select a move without concrete calculation or clear positional justification.

GAME CONTEXT:
- Current Board (FEN): {fen}
- Game History (PGN Movetext): {pgn_movetext}
- Your Turn: Black
- Available Legal Moves (SAN): {', '.join(legal_moves_san)}

REQUIRED ACTION:
After your rigorous internal analysis, identify the single strongest move.
Respond with *ONLY* the chosen move in Standard Algebraic Notation (SAN). No explanations, apologies, or any other text. Just the move.

Example Response: Qxb2
"""
    # --- END OF REVISED PROMPT ---

    print("Requesting LLM move (using demanding analysis prompt with PGN)...")
    try:
        attempts = 0
        max_attempts = 3
        llm_move_san = None
        while attempts < max_attempts:
            attempts += 1
            print(f"LLM Move Attempt {attempts}/{max_attempts}")

            generation_config = genai.types.GenerationConfig(
                max_output_tokens=10,
                temperature=0.35 # Slightly higher temperature
            )

            response = genai_model.generate_content(prompt, generation_config=generation_config)
            candidate_san = response.text.strip()
            print(f"LLM Raw Response: '{candidate_san}'")

            if not candidate_san:
                 print("LLM returned empty response."); time.sleep(0.5); continue

            potential_move = None
            parts = candidate_san.split()
            for part in parts:
                cleaned_part = part.rstrip('.,;!?"\'')
                if chess.SAN_REGEX.match(cleaned_part):
                    potential_move = cleaned_part
                    print(f"Potential SAN found: '{potential_move}' from part '{part}'")
                    break

            if not potential_move:
                print(f"LLM response '{candidate_san}' doesn't contain recognizable SAN."); time.sleep(0.5); continue

            try:
                move = current_board.parse_san(potential_move)
                if move in current_board.legal_moves:
                    llm_move_san = potential_move
                    print(f"LLM proposed valid move: {llm_move_san}")
                    break # Success!
                else:
                    print(f"LLM proposed illegal move: {potential_move} (parsed from '{candidate_san}')")
            except ValueError as e_parse:
                 if "ambiguous" in str(e_parse).lower():
                     print(f"LLM proposed ambiguous SAN: {potential_move}. Failing attempt.")
                 else:
                      print(f"LLM proposed invalid SAN: {potential_move} (Error: {e_parse})")
            except Exception as e_other:
                 print(f"Unexpected error parsing SAN '{potential_move}': {e_other}")

            time.sleep(0.5)

        if llm_move_san:
            return llm_move_san
        else:
            print("LLM failed to provide valid move, choosing random.")
            return random.choice(legal_moves_san) if legal_moves_san else None

    except Exception as e:
        print(f"Error calling Gemini API for move: {e}")
        traceback.print_exc()
        print("Choosing random move due to API error.")
        return random.choice(legal_moves_san) if legal_moves_san else None


# --- LLM Helper for Piece Perspective (No change from previous version) ---
def get_llm_piece_perspective(fen, history_san, piece_code, square, question):
    # (Keep the revised prompt asking for explanations from the previous step)
    if not genai_model:
        return "The connection to my thoughts is unavailable right now (LLM disabled)."

    piece_map = { 'P': 'Pawn', 'N': 'Knight', 'B': 'Bishop', 'R': 'Rook', 'Q': 'Queen', 'K': 'King' }
    piece_type = piece_map.get(piece_code[1].upper(), 'Unknown Piece')
    piece_color = "White"
    pgn_movetext = format_history_as_pgn_movetext(history_san)

    # --- REVISED PROMPT ASKING FOR EXPLANATIONS (FROM PREVIOUS STEP) ---
    prompt = f"""You are the {piece_color} {piece_type} currently positioned on square {square} in a game of chess.
You are acting as an advisor to the player (White). Analyze the current situation from your specific perspective on the board, considering the game's history, and answer the user's question insightfully. Speak in the first person as the piece.

Your Analysis Should Consider:
*   Your potential moves and their consequences (attacks, defenses, squares controlled).
*   Immediate threats against you or nearby friendly pieces.
*   Threats you exert on opponent pieces.
*   Your strategic role (e.g., controlling key squares, defending the king, part of an attack).
*   How the game flow ({pgn_movetext}) influenced your current position and role.

Answering the User's Question:
*   Address the user's question directly and clearly.
*   **If the user asks for advice, strategy, 'what to do', or 'best moves':** Suggest one or two strong candidate moves from White's perspective *and provide a brief (1-2 sentence) explanation* for each, focusing on the strategic or tactical reason (e.g., "Moving to e4 (Nf3) seems good because it controls the center and eyes d5." or "Perhaps Be3? It develops a piece and defends the c5 pawn.").
*   **For other questions:** Answer factually based on your perspective.
*   Keep your overall response concise and focused.

Current board position (FEN): {fen}
Game history (PGN Movetext): {pgn_movetext}

The user (playing White) asks you, the {piece_type} on {square}:
"{question}"

Respond *directly* with your answer as the piece. Do not add introductory phrases like "Okay, as the Queen..." or sign off.
"""
    # --- END OF REVISED PROMPT ---

    print(f"Requesting LLM piece perspective (Piece: {piece_color} {piece_type} on {square}, using PGN, expects explanation)...")
    try:
        generation_config = genai.types.GenerationConfig(
            temperature=0.7
        )
        response = genai_model.generate_content(prompt, generation_config=generation_config)
        answer = response.text.strip()

        prefixes_to_remove = [ f"As the {piece_type} on {square},", f"Okay, speaking as the {piece_type} on {square}:", "Okay, here's my perspective:", "Alright, here's what I see:", f"I am the {piece_type} on {square}." ]
        original_answer = answer
        for prefix in prefixes_to_remove:
            if answer.lower().startswith(prefix.lower()):
                 answer = answer[len(prefix):].lstrip(" ,:.-")

        if answer != original_answer: print(f"LLM Piece Response (Prefix Removed): {answer}")
        else: print(f"LLM Piece Response: {answer}")

        return answer if answer else "I seem to be drawing a blank right now."

    except Exception as e:
        print(f"Error calling Gemini API for piece perspective: {e}")
        traceback.print_exc()
        return f"My thoughts are clouded... (API Error)"


# --- API Endpoints ---
@app.route('/api/game', methods=['GET'])
def get_game_state():
    global board, game_history_san
    winner = None
    termination = None
    is_game_over = board.is_game_over(claim_draw=True)
    if is_game_over:
        outcome = board.outcome(claim_draw=True)
        if outcome:
            winner = "white" if outcome.winner == chess.WHITE else "black" if outcome.winner == chess.BLACK else "draw"
            termination = outcome.termination.name.capitalize().replace('_', ' ')
    if not game_history_san and board.move_stack:
         print("Detected empty SAN history but non-empty move stack, resetting board state for /api/game.")
         board = chess.Board()
    return jsonify({
        'fen': board.fen(), 'turn': 'white' if board.turn == chess.WHITE else 'black',
        'is_game_over': is_game_over, 'winner': winner, 'termination': termination,
        'history': game_history_san
    })

# --- MODIFIED make_move Endpoint ---
@app.route('/api/move', methods=['POST'])
def make_move():
    global board, game_history_san
    if board.turn != chess.WHITE: return jsonify({'error': 'Not White\'s turn'}), 400
    if board.is_game_over(claim_draw=True): return jsonify({'error': 'Game is already over'}), 400

    data = request.get_json()
    uci_move_str = data.get('move')
    if not uci_move_str or not (4 <= len(uci_move_str) <= 5):
        return jsonify({'error': f'Invalid or missing UCI move: {uci_move_str}'}), 400

    opponent_san_response = None # Will hold Lichess or Gemini move SAN
    move_source = None # Track 'Lichess', 'Gemini', 'Random'

    try:
        move = board.parse_uci(uci_move_str)
        if move in board.legal_moves:
            san_move = board.san(move)
            board.push(move) # Push player move
            game_history_san.append(san_move)
            print(f"User (White) played: {uci_move_str} ({san_move})")
            print(f"  Ply Count: {len(game_history_san)}") # Log ply count
            print(f"  New FEN: {board.fen()}")

            if not board.is_game_over(claim_draw=True):
                # --- Decide Black's Move Source ---
                if len(game_history_san) < OPENING_MOVE_LIMIT:
                    print(f"Attempting Lichess move (Ply < {OPENING_MOVE_LIMIT})...")
                    opponent_san = get_lichess_opening_move(board)
                    if opponent_san:
                        opponent_san_response = opponent_san
                        move_source = "Lichess"
                    else:
                        # Fallback to Gemini if Lichess fails
                        print("Lichess failed or returned no move, falling back to Gemini...")
                        opponent_san_response = get_llm_move(board, game_history_san)
                        move_source = "Gemini (Lichess Fallback)"
                else:
                    # Use Gemini after opening limit
                    print(f"Attempting Gemini move (Ply >= {OPENING_MOVE_LIMIT})...")
                    opponent_san_response = get_llm_move(board, game_history_san)
                    move_source = "Gemini"

                # If still no move (e.g., Gemini also failed), try random as last resort
                if not opponent_san_response:
                     print(f"{move_source or 'Move generation'} failed, trying random move...")
                     legal_moves = [board.san(m) for m in board.legal_moves]
                     if legal_moves:
                         opponent_san_response = random.choice(legal_moves)
                         move_source = "Random (Fallback)"
                     else:
                          print("No legal moves available for Black?") # Should be caught by game over check

                # --- Push Opponent's Move (if found) ---
                if opponent_san_response:
                    try:
                        llm_move = board.parse_san(opponent_san_response)
                        if llm_move in board.legal_moves:
                             board.push(llm_move) # Push opponent move
                             game_history_san.append(opponent_san_response)
                             print(f"Opponent ({move_source}) played: {opponent_san_response}")
                             print(f"  Ply Count: {len(game_history_san)}")
                             print(f"  New FEN: {board.fen()}")
                        else:
                             print(f"CRITICAL BACKEND ERROR: {move_source} move {opponent_san_response} validated but illegal?")
                    except ValueError as e_parse_opp:
                         print(f"CRITICAL BACKEND ERROR: {move_source} move {opponent_san_response} failed SAN parsing? Error: {e_parse_opp}")
                         traceback.print_exc()
                    except Exception as e_push_opp:
                         print(f"Error processing {move_source} move {opponent_san_response}: {e_push_opp}")
                         traceback.print_exc()

            # --- Return final state ---
            final_state_data = get_game_state().get_json()
            # Add source info if needed, but response SAN is already opponent_san_response
            final_state_data['last_llm_move'] = opponent_san_response # Keep key name for frontend consistency
            return jsonify(final_state_data)

        else: # User move was illegal
             try: illegal_san = board.san(move)
             except: illegal_san = uci_move_str
             print(f"Illegal user move attempt: {uci_move_str} ({illegal_san})")
             return jsonify({'error': f'Illegal move: {illegal_san} ({uci_move_str})'}), 400

    except ValueError as e_parse_user: # Catches user's bad UCI
        print(f"Invalid user move notation: {uci_move_str} - Error: {e_parse_user}")
        return jsonify({'error': f'Invalid or illegal move notation: {uci_move_str}'}), 400
    except Exception as e:
        print(f"Unexpected error during move processing: {e}")
        traceback.print_exc()
        return jsonify({'error': 'An internal server error occurred during move processing'}), 500


@app.route('/api/reset', methods=['POST'])
def reset_game():
    global board, game_history_san
    board = chess.Board()
    game_history_san = []
    print("Game reset.")
    state_data = get_game_state().get_json()
    return jsonify(state_data)


@app.route('/api/ask_piece', methods=['POST'])
def ask_piece():
    global game_history_san
    if not genai_model: return jsonify({'error': 'LLM not configured.'}), 503
    data = request.get_json()
    question, piece_code, square, fen = data.get('question'), data.get('piece'), data.get('square'), data.get('fen')
    current_history = game_history_san
    if not all([question, piece_code, square, fen]): return jsonify({'error': 'Missing required data'}), 400
    if not (piece_code.startswith('w') and len(piece_code) == 2 and piece_code[1] in 'PNBRQK'): return jsonify({'error': f'Invalid piece code: {piece_code}.'}), 400
    try: chess.parse_square(square); chess.Board(fen)
    except ValueError: return jsonify({'error': f'Invalid square or FEN'}), 400
    try:
        answer = get_llm_piece_perspective(fen, current_history, piece_code, square, question)
        return jsonify({'answer': answer})
    except Exception as e:
        print(f"Error in /api/ask_piece route: {e}"); traceback.print_exc()
        return jsonify({'error': 'Internal server error asking piece.'}), 500


# --- Frontend Serving Route (No changes needed here) ---
@app.route('/')
def index():
    # HTML content remains the same as in the previous successful version
    html_content = """
<!DOCTYPE html>
<html lang="en">
<head>
    <meta charset="UTF-8">
    <meta name="viewport" content="width=device-width, initial-scale=1.0">
    <title>Chess vs Gemini/Lichess (Talk to Pieces)</title> <!-- Updated Title -->

    <!-- Chessboard.js CSS -->
    <link rel="stylesheet"
          href="https://unpkg.com/@chrisoakman/chessboardjs@1.0.0/dist/chessboard-1.0.0.min.css"
          crossorigin="anonymous">

    <!-- Include jQuery -->
    <script src="https://code.jquery.com/jquery-3.6.0.min.js"
            integrity="sha256-/xUj+3OJU5yExlq6GSYGSHk7tPXikynS7ogEvDej/m4="
            crossorigin="anonymous"></script>

    <!-- Chessboard.js JS -->
    <script src="https://unpkg.com/@chrisoakman/chessboardjs@1.0.0/dist/chessboard-1.0.0.min.js"
            crossorigin="anonymous"></script>

    <style>
        body { font-family: sans-serif; display: flex; justify-content: center; padding: 20px; background-color: #f0f0f0; }
        .container { display: flex; flex-direction: column; align-items: center; background-color: #fff; padding: 20px; border-radius: 8px; box-shadow: 0 2px 10px rgba(0,0,0,0.1); max-width: 500px; /* Limit width */}
        #myBoard { width: 90vw; max-width: 450px; /* Responsive board */ margin-bottom: 15px; }
        .status-info { font-size: 1.1em; font-weight: bold; margin-bottom: 10px; padding: 8px 15px; border-radius: 4px; text-align: center; min-width: 250px; }
        .status-info.turn-white { background-color: #e0ffe0; border: 1px solid #90ee90; color: #006400; }
        .status-info.turn-black { background-color: #f0f0f0; border: 1px solid #cccccc; color: #333; }
        .status-info.thinking { background-color: #fffacd; border: 1px solid #ffd700; color: #b8860b; animation: pulse 1.5s infinite ease-in-out; }
        .status-info.game-over { background-color: #add8e6; border: 1px solid #87ceeb; color: #00008b; }
        .status-info.error { background-color: #ffe4e1; border: 1px solid #ffb6c1; color: #dc143c; }
        .controls { margin-bottom: 15px; }
        .button { padding: 10px 15px; font-size: 1em; border: none; border-radius: 4px; cursor: pointer; margin: 0 5px; transition: background-color 0.2s; }
        .button:disabled { background-color: #ccc; cursor: not-allowed; }
        .button-primary { background-color: #4CAF50; color: white; }
        .button-primary:hover:not(:disabled) { background-color: #45a049; }
        .button-secondary { background-color: #008CBA; color: white; }
        .button-secondary:hover:not(:disabled) { background-color: #007ba7; }
        .history-log { max-height: 150px; overflow-y: auto; border: 1px solid #ccc; padding: 10px; width: 90%; margin-top: 10px; background-color: #f9f9f9; border-radius: 4px; }
        .history-log h2 { margin: 0 0 10px 0; font-size: 1.1em; text-align: center; border-bottom: 1px solid #eee; padding-bottom: 5px; color: #555; }
        #history-list { list-style: none; padding: 0; margin: 0; font-size: 0.9em; }
        #history-list li { padding: 3px 0; border-bottom: 1px dotted #eee; display: flex; justify-content: space-between; }
        #history-list li:last-child { border-bottom: none; }
        #history-list .move-number { color: #888; font-weight: bold; flex-basis: 10%; text-align: right; padding-right: 5px;}
        #history-list .move-white { flex-basis: 45%; text-align: center; font-weight: 500;}
        #history-list .move-black { flex-basis: 45%; text-align: left; font-weight: normal;}


        /* --- Dialog Box Styles --- */
        .dialog-overlay { position: fixed; top: 0; left: 0; width: 100%; height: 100%; background-color: rgba(0, 0, 0, 0.6); display: none; justify-content: center; align-items: center; z-index: 1000; }
        .dialog-box { background-color: white; padding: 25px; border-radius: 8px; box-shadow: 0 5px 15px rgba(0,0,0,0.3); text-align: center; max-width: 400px; width: 90%; position: relative; }
        .dialog-box h3 { margin-top: 0; margin-bottom: 15px; font-size: 1.2em; color: #333; }
        .dialog-box textarea { width: 95%; min-height: 60px; margin-bottom: 15px; padding: 8px; border: 1px solid #ccc; border-radius: 4px; font-size: 1em; resize: vertical; }
        .dialog-box .dialog-buttons { display: flex; justify-content: space-around; }
        .dialog-response { margin-top: 15px; padding: 10px; background-color: #eef; border: 1px solid #ccd; border-radius: 4px; text-align: left; font-style: italic; max-height: 150px; overflow-y: auto; min-height: 40px; }
        .dialog-response.thinking { color: #888; display: flex; align-items: center; justify-content: center;}
        .dialog-response.error { color: #dc143c; font-weight: bold; }
        .dialog-close-btn { position: absolute; top: 10px; right: 10px; background: none; border: none; font-size: 1.5em; cursor: pointer; color: #888; line-height: 1; padding: 0;}
        .dialog-close-btn:hover { color: #333; }

        /* Animation for thinking status */
        @keyframes pulse { 0% { opacity: 1; } 50% { opacity: 0.6; } 100% { opacity: 1; } }
    </style>
</head>
<body>
    <div class="container">
        <h1>Chess vs Gemini/Lichess</h1> <!-- Updated Title -->
        <p>You play White. Opponent uses Lichess openings (first 5 moves), then Gemini. Right-click your pieces to talk to them!</p> <!-- Updated Description -->
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

       <script>
        // Frontend Javascript remains the same as the previous working version.
        // No changes are needed here as the logic swap (Lichess vs Gemini)
        // happens entirely on the backend within the /api/move endpoint.
        // The frontend just sends the move and receives the updated game state.
        $(document).ready(function() {
            console.log("JS: Document ready.");

            // 1. Variable Declarations
            var board = null; // Chessboard.js instance
            var currentFen = 'start';
            var currentGameHistory = []; // Stores SAN moves received from backend
            var isPlayerTurn = true;
            var isGameOver = false;
            var isProcessingMove = false; // Flag to prevent interaction during processing
            var $status = $('#status');
            var $historyList = $('#history-list');
            var $boardElement = $('#myBoard'); // The div Chessboard is attached to
            var $dialogOverlay = $('#piece-dialog-overlay');
            var $dialogTitle = $('#dialog-title');
            var $dialogQuestion = $('#dialog-question');
            var $dialogResponseArea = $('#dialog-response-area');
            var $dialogSubmit = $('#dialog-submit');
            var $dialogCancel = $('#dialog-cancel');
            var $dialogClose = $('#dialog-close');
            var currentDialogData = null; // Stores {piece, square} for the open dialog
            var isRightMouseButtonDown = false; // Flag for right mouse button detection

            const API_BASE_URL = "/api";

            // 2. Helper Function Definitions
            async function fetchApi(endpoint, method = 'GET', body = null) {
                const url = `${API_BASE_URL}${endpoint}`;
                const options = {
                    method: method,
                    headers: { 'Content-Type': 'application/json', 'Accept': 'application/json' }
                };
                if (body) { options.body = JSON.stringify(body); }
                console.log(`JS: Fetching ${method} ${url}`, body ? JSON.stringify(body) : '');
                try {
                    const response = await fetch(url, options);
                    const responseData = await response.json(); // Assume JSON response
                    console.log(`JS: Fetch response for ${method} ${url} - Status: ${response.status}`, responseData);
                    if (!response.ok) {
                         const errorMessage = responseData.error || `HTTP error! Status: ${response.status}`;
                         console.error(`API Error Response (${method} ${url}):`, responseData);
                         throw new Error(errorMessage);
                    }
                    return responseData;
                } catch (error) {
                    console.error(`API Error Caught (${method} ${url}):`, error);
                    updateStatus(`Error: ${error.message || 'Network or server error'}`, true);
                    throw error; // Re-throw
                }
            }

            function updateStatus(message, isError = false, isThinking = false, baseClass = null) {
                if(!$status.length) { console.error("JS: Status element #status not found!"); return; }
                console.log(`JS: updateStatus - Message: "${message}", Error: ${isError}, Thinking: ${isThinking}, BaseClass: ${baseClass}`);
                $status.text(message);
                $status.removeClass('error thinking turn-white turn-black game-over'); // Clear previous states
                if(baseClass) $status.addClass(baseClass);
                if(isError) $status.addClass('error');
                if(isThinking) $status.addClass('thinking');
            }

             function updateHistoryList(history) {
                 if(!$historyList.length) { console.warn("JS: History list element #history-list not found."); return; }
                 $historyList.empty();
                 currentGameHistory = Array.isArray(history) ? history : [];
                 console.log("JS: Updating history list with:", currentGameHistory);

                 if (currentGameHistory.length === 0) {
                     $historyList.append('<li>No moves yet.</li>'); return;
                 }

                 for (let i = 0; i < currentGameHistory.length; i += 2) {
                     const moveNumber = Math.floor(i / 2) + 1;
                     const whiteMove = currentGameHistory[i] || '';
                     const blackMove = currentGameHistory[i + 1] || (isGameOver ? '' : '...');

                     const listItem = `<li>
                         <span class="move-number">${moveNumber}.</span>
                         <span class="move-white">${whiteMove}</span>
                         <span class="move-black">${blackMove}</span>
                     </li>`;
                     $historyList.append(listItem);
                 }
                 const historyLogElement = $historyList.closest('.history-log')[0];
                 if (historyLogElement) { historyLogElement.scrollTop = historyLogElement.scrollHeight; }
            }


             function getPieceName(pieceCode) {
                const map = { 'P': 'Pawn', 'N': 'Knight', 'B': 'Bishop', 'R': 'Rook', 'Q': 'Queen', 'K': 'King' };
                if (typeof pieceCode === 'string' && pieceCode.length === 2) {
                    return map[pieceCode[1].toUpperCase()] || 'Unknown Piece';
                }
                return 'Unknown Piece';
            }


            // 3. State Update Function (No changes needed from previous version)
            function updateGameState(data) {
                console.log("JS: updateGameState called with data:", data);
                if (data && typeof data.fen === 'string') {
                    console.log("JS: updateGameState - Data is valid, processing state.");
                    const wasLoading = $status.text() === "Loading game...";

                    currentFen = data.fen;
                    currentGameHistory = Array.isArray(data.history) ? data.history : [];
                    isGameOver = data.is_game_over || false;
                    isPlayerTurn = (data.turn === 'white') && !isGameOver;

                    let statusMsg = "Status Error"; let statusClass = "error";
                    if (isGameOver) {
                        statusClass = "game-over";
                        const terminationReason = data.termination ? `(${data.termination})` : '';
                        if (data.winner === 'white') statusMsg = `🎉 Game Over! You Won! ${terminationReason}`;
                        else if (data.winner === 'black') statusMsg = `🤖 Game Over! Opponent Won! ${terminationReason}`; // Changed LLM to Opponent
                        else statusMsg = `🤝 Game Over! Draw! ${terminationReason}`;
                    } else if (isPlayerTurn) {
                        statusClass = "turn-white"; statusMsg = "⚪ Your Turn (White)";
                    } else {
                        statusClass = "turn-black"; statusMsg = "⚫ Opponent is thinking..."; // Changed LLM to Opponent
                        if (!wasLoading) { updateStatus(statusMsg, false, true, statusClass); }
                    }
                    console.log(`JS: updateGameState - Status determined: "${statusMsg}", Class: ${statusClass}`);

                    const finalStatusMsg = statusMsg;
                    const finalStatusClass = statusClass;
                    const isThinkingStatus = (finalStatusClass === 'turn-black' && !isGameOver && !wasLoading);

                    const updateFunc = () => { updateStatus(finalStatusMsg, false, isThinkingStatus, finalStatusClass); };

                    if (wasLoading) { setTimeout(updateFunc, 150); }
                    else { updateFunc(); }

                    updateHistoryList(currentGameHistory);

                    if (board && typeof board.position === 'function') {
                        board.position(currentFen, false);
                    } else { console.error("JS: updateGameState - CRITICAL: Board object invalid!"); }

                    if(data.last_llm_move) { // Check if opponent moved
                         console.log(`JS: updateGameState detected opponent move: ${data.last_llm_move}`);
                         if (!isGameOver) { updateStatus("⚪ Your Turn (White)", false, false, "turn-white"); }
                    } else if (!isPlayerTurn && !isGameOver && !isThinkingStatus) {
                         updateStatus("⚫ Opponent's Turn (Black)", false, false, "turn-black"); // Changed LLM to Opponent
                    }

                } else {
                     console.error("JS: updateGameState received invalid data:", data);
                     updateStatus("Error: Invalid game state from server.", true);
                     isPlayerTurn = false; isGameOver = true;
                }
                console.log("JS: updateGameState finished.");
            }


            // 4. Event Handler Definitions (No changes needed from previous version)
            function onBoardMouseDown(event) {
                if (event.button === 2) {
                    isRightMouseButtonDown = true; event.preventDefault();
                } else if (event.button === 0 || event.button === 1) {
                    isRightMouseButtonDown = false;
                }
            }
            $boardElement.on('contextmenu', (e) => e.preventDefault());

            function onDragStart (source, piece, position, orientation) {
                if (isRightMouseButtonDown) {
                    if (piece.search(/^w/) === -1) { isRightMouseButtonDown = false; return false; }
                    return true; // Allow right-drag start on white pieces
                }
                // Prevent left-drag if invalid state
                if (isGameOver || !isPlayerTurn || piece.search(/^b/) !== -1 || isProcessingMove) {
                    return false;
                }
                return true; // Allow left-drag
            }

            async function onDrop (source, target, piece, newPos, oldPos, orientation) {
                const wasRightMouseDown = isRightMouseButtonDown;
                isRightMouseButtonDown = false; // Reset flag

                if (wasRightMouseDown && source === target) { // Right Click Release
                    if (!piece || piece.search(/^w/) === -1 || isProcessingMove) return;
                    openPieceDialog(piece, source); return;
                }
                if (wasRightMouseDown && source !== target) return 'snapback'; // Right Drag
                if (source === target) return 'snapback'; // Left Click (No Drag)
                if (isGameOver || !isPlayerTurn || isProcessingMove) return 'snapback'; // Invalid state for move

                // --- Actual Move (Left Drag Drop) ---
                var moveUCI = source + target;
                if (piece === 'wP' && target.charAt(1) === '8') { moveUCI += 'q'; }

                isProcessingMove = true;
                $('#reset-game').prop('disabled', true);
                updateStatus("⚪ Sending your move...", false, true, "turn-white");

                try {
                    const data = await fetchApi('/move', 'POST', { move: moveUCI });
                    updateGameState(data);
                } catch (error) {
                    if (board) board.position(currentFen, false); // Revert visual board
                    return 'snapback'; // Return snapback on error
                } finally {
                     isProcessingMove = false;
                     $('#reset-game').prop('disabled', isGameOver);
                }
            }


            // 5. Dialog Logic Functions (No changes needed from previous version)
            function openPieceDialog(piece, square) {
                if (!piece || !square) return;
                currentDialogData = { piece, square };
                $dialogTitle.text(`Ask the ${getPieceName(piece)} on ${square}`);
                $dialogQuestion.val('');
                $dialogResponseArea.hide().empty().removeClass('error thinking');
                $dialogSubmit.prop('disabled', false);
                $dialogCancel.prop('disabled', false);
                $dialogOverlay.css('display', 'flex');
                $dialogQuestion.focus();
            }
            function closePieceDialog() { $dialogOverlay.hide(); currentDialogData = null; }
            async function submitPieceQuestion() {
                if (!currentDialogData) return;
                const question = $dialogQuestion.val().trim();
                if (!question) {
                     $dialogResponseArea.text('Please enter a question.').addClass('error').show();
                     setTimeout(() => { if (!$dialogResponseArea.hasClass('thinking')) $dialogResponseArea.hide().removeClass('error'); }, 2000);
                     return;
                 }
                const { piece, square } = currentDialogData;
                $dialogResponseArea.html('<i>Thinking...</i>').addClass('thinking').removeClass('error').show();
                $dialogSubmit.prop('disabled', true); $dialogCancel.prop('disabled', true);
                try {
                    const data = await fetchApi('/ask_piece', 'POST', { question, piece, square, fen: currentFen });
                    $dialogResponseArea.text(data.answer || 'Received empty answer.').removeClass('thinking error');
                } catch (error) {
                     $dialogResponseArea.text(`Error: ${error.message || 'Could not get answer.'}`).addClass('error').removeClass('thinking');
                } finally {
                     $dialogSubmit.prop('disabled', false); $dialogCancel.prop('disabled', false);
                }
            }

            // 6. Initialization Function (No changes needed from previous version)
            async function initializeGame() {
                updateStatus("Loading game...", false, true);
                isProcessingMove = true; $('#reset-game').prop('disabled', true);
                try {
                    const data = await fetchApi('/game');
                    if (!data || typeof data.fen !== 'string' || !Array.isArray(data.history)) { throw new Error("Invalid initial game data."); }
                    currentFen = data.fen; currentGameHistory = data.history;
                    var config = {
                        draggable: true, position: currentFen, orientation: 'white',
                        onDragStart: onDragStart, onDrop: onDrop,
                        pieceTheme: '/static/{piece}.png',
                        moveSpeed: 300, snapbackSpeed: 500, snapSpeed: 100
                    };
                    if (!$('#myBoard').length) { throw new Error("#myBoard element not found."); }
                    board = Chessboard('myBoard', config);
                    if (!board || typeof board.position !== 'function') { throw new Error("Chessboard init failed."); }
                    $(window).resize(() => { if(board) board.resize(); }).trigger('resize');
                    $boardElement.on('mousedown', onBoardMouseDown);
                    updateGameState(data);
                } catch (error) {
                    updateStatus(`Initialization Failed: ${error.message || 'Unknown error'}`, true);
                } finally {
                    isProcessingMove = false; $('#reset-game').prop('disabled', isGameOver);
                }
            }

            // 7. Event Listener Bindings (No changes needed from previous version)
             $('#reset-game').on('click', async function() {
                 if (isProcessingMove) return;
                 isProcessingMove = true; $(this).prop('disabled', true);
                 closePieceDialog(); updateStatus("🔄 Resetting game...", false, true);
                 try {
                    const data = await fetchApi('/reset', 'POST');
                    if (board) board.position('start', false);
                    updateGameState(data);
                 } catch (error) { /* Status updated by fetchApi */ }
                 finally { isProcessingMove = false; $(this).prop('disabled', false); }
             });
            $dialogSubmit.on('click', submitPieceQuestion);
            $dialogCancel.on('click', closePieceDialog); $dialogClose.on('click', closePieceDialog);
            $dialogQuestion.on('keypress', (e) => { if (e.key==='Enter' && !e.shiftKey) { e.preventDefault(); submitPieceQuestion(); }});
            $(document).on('keydown', (e) => { if (e.key === "Escape" && $dialogOverlay.is(':visible')) { closePieceDialog(); }});

            // 8. Start Application
            initializeGame();

        }); // End of $(document).ready
       </script>

</body>
</html>
    """
    return Response(html_content, mimetype='text/html')

# --- Main Execution ---
if __name__ == '__main__':
    print("--- Starting Flask Server ---")
    print(f"API Key Loaded: {'Yes' if GEMINI_API_KEY else 'No'}")
    print(f"Gemini Model: {MODEL_NAME if genai_model else 'Not Initialized'}")
    print(f"Opponent Moves: Lichess API for first {OPENING_MOVE_LIMIT} plies, then {MODEL_NAME}")
    print("Features: Talk to White Pieces (Revised Prompt w/ Explanation)")
    print("Access at: http://127.0.0.1:5000/")
    print("-----------------------------")
    # Ensure requests library is installed: pip install requests
    app.run(host='127.0.0.1', port=5000, debug=True, use_reloader=True)
