import os
import chess
import chess.pgn
import google.generativeai as genai
import random
import time
from flask import Flask, request, jsonify, Response
from flask_cors import CORS
from dotenv import load_dotenv

# --- Environment & API Key Setup ----
load_dotenv()
GEMINI_API_KEY = os.getenv("GOOGLE_API_KEY") or os.getenv("CURSOR_GOOGLE_API_KEY")
genai_model = None
if GEMINI_API_KEY:
    try:
        genai.configure(api_key=GEMINI_API_KEY)
        # Use a model suitable for chat/instruction following, Pro might be better if available and affordable
        # Flash is good for speed, but might be less nuanced for analysis. Test and adjust.
        MODEL_NAME = 'gemini-2.0-flash'
        genai_model = genai.GenerativeModel(MODEL_NAME)
        print(f"Gemini Model Initialized ({MODEL_NAME}).")
    except Exception as e:
        print(f"Error initializing Gemini model: {e}. LLM disabled.")
else:
    print("Warning: GOOGLE_API_KEY environment variable not set. LLM disabled.")

# --- Flask App Setup ---
app = Flask(__name__, static_folder='static', static_url_path='/static')
CORS(app, resources={r"/api/*": {"origins": "*"}}) # Allow all origins for simplicity

# --- Game State ---
board = chess.Board()
game_history_san = []

# --- LLM Helper for Black Moves (No change needed) ---
# --- LLM Helper for Black Moves ---
def get_llm_move(current_board, history_san):
    if not genai_model:
        print("LLM not available. Choosing random move.")
        legal_moves = [current_board.san(move) for move in current_board.legal_moves]
        return random.choice(legal_moves) if legal_moves else None

    legal_moves = [current_board.san(move) for move in current_board.legal_moves]
    if not legal_moves: return None

    fen = current_board.fen()
    history_str = " ".join(history_san)

    # --- NEW, MORE DETAILED PROMPT ---
    prompt = f"""You are a powerful chess engine simulating a world-class grandmaster playing as Black. Your objective is to identify and play the objectively strongest possible move in the current position.

BEFORE making a decision, perform a deep internal analysis focusing on these CRITICAL factors:
1.  **King Safety:** Assess any immediate or potential threats to your King. Evaluate the safety of the opponent's King.
2.  **Material Balance:** Calculate the current material count precisely. Identify any hanging (undefended) pieces for both sides.
3.  **Tactical Opportunities:** Search diligently for tactics such as forks, pins, skewers, discovered attacks, removing the defender, and potential checkmates. Do NOT make simple blunders or hang pieces.
4.  **Positional & Strategic Factors:** Evaluate piece activity, control of central squares (e4, d4, e5, d5), open files, pawn structure weaknesses or strengths, and outpost squares for knights.
5.  **Threats:** Identify all immediate threats your opponent poses and all direct threats you can create with your potential moves.

GAME CONTEXT:
- Current Board (FEN): {fen}
- Game History (SAN): {history_str}
- Your Turn: Black
- Available Legal Moves (SAN): {', '.join(legal_moves)}

REQUIRED ACTION:
After your thorough internal analysis based on the principles above, select the single strongest legal move.
Respond with *ONLY* the chosen move in Standard Algebraic Notation (SAN). Do not include any explanation, preamble, or other text.

Example Response: Nf6
"""
    # --- END OF NEW PROMPT ---


    print("Requesting LLM move (using detailed analysis prompt)...")
    try:
        attempts = 0
        max_attempts = 3 # Keep retries in case of temporary API issues or invalid format responses
        llm_move_san = None
        while attempts < max_attempts:
            attempts += 1
            print(f"LLM Move Attempt {attempts}/{max_attempts}")

            # --- ADJUST GENERATION CONFIG ---
            # Lower temperature for more deterministic, less "creative" (potentially weaker) moves.
            # Focus on what the model thinks is the "best" based on training.
            generation_config = genai.types.GenerationConfig(
                max_output_tokens=10, # Still only need the move itself
                temperature=0.2       # <<-- LOWER TEMPERATURE
            )
            # --- END CONFIG ADJUSTMENT ---

            # Ensure you are using the intended model (e.g., gemini-1.5-pro-latest)
            # If you created a separate move_model instance, use that. Otherwise, use genai_model.
            response = genai_model.generate_content(prompt, generation_config=generation_config) # Or move_model.generate_content(...)

            candidate_san = response.text.strip()
            print(f"LLM Raw Response: '{candidate_san}'")

            if not candidate_san:
                 print("LLM returned empty response.")
                 continue

            potential_move = None
            parts = candidate_san.split()
            for part in parts:
                 cleaned_part = part.rstrip('.,;!?')
                 # Use a more robust regex check if needed, but SAN_REGEX is usually good
                 if chess.SAN_REGEX.match(cleaned_part):
                      potential_move = cleaned_part
                      break

            if not potential_move:
                print(f"LLM response '{candidate_san}' doesn't contain recognizable SAN.")
                time.sleep(0.5) # Wait a bit before retrying on format error
                continue

            try:
                move = current_board.parse_san(potential_move)
                if move in current_board.legal_moves:
                    llm_move_san = potential_move
                    print(f"LLM proposed valid move: {llm_move_san}")
                    break # Success!
                else:
                    print(f"LLM proposed illegal move: {potential_move} (parsed from '{candidate_san}')")
            except ValueError:
                print(f"LLM proposed invalid SAN: {potential_move} (parsed from '{candidate_san}')")

            time.sleep(0.5) # Small delay before retry if move was invalid/illegal

        if llm_move_san:
            return llm_move_san
        else:
            print("LLM failed to provide a valid move after multiple attempts, choosing random move.")
            return random.choice(legal_moves) if legal_moves else None

    except Exception as e:
        print(f"Error calling Gemini API for move: {e}")
        import traceback
        traceback.print_exc() # Print full traceback for API errors
        print("Choosing random move due to API error.")
        return random.choice(legal_moves) if legal_moves else None

# --- NEW: LLM Helper for Piece Perspective ---
def get_llm_piece_perspective(fen, history_san, piece_code, square, question):
    if not genai_model:
        return "The connection to my thoughts is unavailable right now (LLM disabled)."

    # Map piece code to a more natural name
    piece_map = {
        'P': 'Pawn', 'N': 'Knight', 'B': 'Bishop',
        'R': 'Rook', 'Q': 'Queen', 'K': 'King'
    }
    # Assumes piece_code is like 'wP', 'bN' etc. We only care about white pieces here.
    piece_type = piece_map.get(piece_code[1].upper(), 'Unknown Piece')
    piece_color = "White" # User is always white

    history_str = " ".join(history_san)

    # Construct the prompt carefully
    prompt = f"""You are the {piece_color} {piece_type} currently positioned on square {square} in a game of chess.
Your task is to analyze the current situation from your specific perspective on the board and answer the user's question.
Focus on your potential moves, threats against you, threats you exert, your strategic importance, and how you contribute to White's game.
Be insightful but keep your answer concise and directly address the question. Speak in the first person as the piece.

Current board position (FEN): {fen}
Game history (SAN): {history_str}

The user (playing White) asks you, the {piece_type} on {square}:
"{question}"

Respond *only* with your answer as the piece. Do not add introductory phrases like "As the Queen..." or sign off.
"""

    print(f"Requesting LLM piece perspective (Piece: {piece_color} {piece_type} on {square})...")
    try:
        # Use a model potentially better suited for conversational analysis if available
        # Using the main configured model here. Adjust temp/tokens as needed.
        generation_config = genai.types.GenerationConfig(
            # max_output_tokens=150, # Allow for a slightly longer, more detailed answer
            temperature=0.7      # Allow for a bit more creativity/natural language
        )
        response = genai_model.generate_content(prompt, generation_config=generation_config)
        answer = response.text.strip()
        print(f"LLM Piece Response: {answer}")
        return answer if answer else "I'm drawing a blank right now."

    except Exception as e:
        print(f"Error calling Gemini API for piece perspective: {e}")
        return f"My thoughts are clouded... (API Error: {e})"

# --- API Endpoints ---
@app.route('/api/game', methods=['GET'])
def get_game_state():
    global board, game_history_san
    winner = None
    termination = None
    is_game_over = board.is_game_over(claim_draw=True)
    if is_game_over:
        game_outcome = board.outcome(claim_draw=True)
        if game_outcome:
            winner = "white" if game_outcome.winner == chess.WHITE else "black" if game_outcome.winner == chess.BLACK else "draw"
            termination = game_outcome.termination.name.capitalize()

    # Ensure board state is fresh if history is empty and board is not initial
    if not game_history_san and board.fen() != chess.STARTING_FEN:
         board = chess.Board()
         print("Detected empty history and non-start board, resetting board state for /api/game.")

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
    global board, game_history_san
    if board.turn != chess.WHITE:
        return jsonify({'error': 'Not White\'s turn'}), 400
    if board.is_game_over(claim_draw=True):
        return jsonify({'error': 'Game is already over'}), 400

    data = request.get_json()
    uci_move_str = data.get('move') # Expect UCI like "e2e4" or "e7e8q"
    if not uci_move_str:
        return jsonify({'error': 'Move (UCI format) not provided'}), 400
    if not (4 <= len(uci_move_str) <= 5): # Basic UCI length check
        return jsonify({'error': f'Invalid UCI format: {uci_move_str}'}), 400

    llm_san_response = None
    try:
        move = board.parse_uci(uci_move_str)
        if move in board.legal_moves:
            san_move = board.san(move)
            game_history_san.append(san_move)
            board.push(move)
            print(f"User (White) played: {uci_move_str} ({san_move})")

            if not board.is_game_over(claim_draw=True):
                # --- Get LLM Move ---
                llm_san = get_llm_move(board, game_history_san)
                if llm_san:
                    try:
                        llm_move = board.parse_san(llm_san)
                        # Double check legality just in case LLM hallucinated a valid SAN for an illegal move
                        if llm_move in board.legal_moves:
                             board.push(llm_move)
                             game_history_san.append(llm_san)
                             llm_san_response = llm_san
                             print(f"LLM (Black) played: {llm_san}")
                        else:
                             print(f"CRITICAL BACKEND ERROR: LLM move {llm_san} validated but illegal on board {board.fen()}?")
                             # Fallback: Maybe try random again or return error? For now, just log.
                             # We might need to handle this scenario more robustly.
                    except ValueError:
                         print(f"CRITICAL BACKEND ERROR: LLM move {llm_san} validated but parse_san failed?")
                         # Handle error, maybe fallback
                    except Exception as e:
                         print(f"Error pushing LLM move {llm_san}: {e}")
                         # Handle error

            # Fetch the final state AFTER both moves (or just player's move if game ends)
            final_state_board = board # Use the global board reflecting all moves
            state_data = get_game_state().get_json() # Fetch based on current global board
            state_data['last_llm_move'] = llm_san_response # Add LLM move info if applicable
            return jsonify(state_data)
        else:
             # Find SAN for illegal move attempt if possible for better error message
             try:
                 illegal_san = board.san(move)
             except ValueError: # Move might be impossible (e.g., Knight from e4 to e5)
                 illegal_san = uci_move_str # Fallback to UCI
             return jsonify({'error': f'Illegal move: {illegal_san} ({uci_move_str})'}), 400

    except ValueError: # Catches errors from parse_uci if format is bad or move impossible from square
        return jsonify({'error': f'Invalid or illegal move notation: {uci_move_str}'}), 400
    except Exception as e:
        print(f"Unexpected error during move processing: {e}")
        # Consider logging the full traceback for debugging
        import traceback
        traceback.print_exc()
        return jsonify({'error': 'An internal server error occurred during move processing'}), 500

@app.route('/api/reset', methods=['POST'])
def reset_game():
    global board, game_history_san
    board = chess.Board()
    game_history_san = []
    print("Game reset.")
    return get_game_state()

# --- NEW: API Endpoint for Asking a Piece ---
@app.route('/api/ask_piece', methods=['POST'])
def ask_piece():
    if not genai_model:
        return jsonify({'error': 'LLM (Gemini) is not configured or available.'}), 503 # Service Unavailable

    data = request.get_json()
    question = data.get('question')
    piece_code = data.get('piece') # e.g., "wP"
    square = data.get('square')   # e.g., "e4"
    fen = data.get('fen')         # Current FEN state
    history = data.get('history') # List of SAN moves

    if not all([question, piece_code, square, fen, history is not None]):
        return jsonify({'error': 'Missing required data: question, piece, square, fen, history'}), 400

    # Basic validation
    if not (piece_code.startswith('w') and len(piece_code) == 2 and piece_code[1] in 'PNBRQK'):
         return jsonify({'error': f'Invalid piece code: {piece_code}. Must be a white piece (wP, wN, etc.).'}), 400
    if not chess.SQUARE_NAMES.__contains__(square):
         return jsonify({'error': f'Invalid square: {square}'}), 400
    # Could add FEN validation if needed, but assume frontend sends valid state

    try:
        answer = get_llm_piece_perspective(fen, history, piece_code, square, question)
        return jsonify({'answer': answer})
    except Exception as e:
        print(f"Error in /api/ask_piece route: {e}")
        import traceback
        traceback.print_exc()
        return jsonify({'error': 'An internal server error occurred while asking the piece.'}), 500


# --- Frontend Serving Route ---
@app.route('/')
def index():
    # Indentation fixed for the HTML string
    html_content = """
<!DOCTYPE html>
<html lang="en">
<head>
    <meta charset="UTF-8">
    <meta name="viewport" content="width=device-width, initial-scale=1.0">
    <title>Chess vs Gemini (Talk to Pieces)</title>

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
        .container { display: flex; flex-direction: column; align-items: center; background-color: #fff; padding: 20px; border-radius: 8px; box-shadow: 0 2px 10px rgba(0,0,0,0.1); }
        #myBoard { width: 400px; /* Adjust as needed */ margin-bottom: 15px; }
        .status-info { font-size: 1.1em; font-weight: bold; margin-bottom: 10px; padding: 8px 15px; border-radius: 4px; text-align: center; min-width: 250px; }
        .status-info.turn-white { background-color: #e0ffe0; border: 1px solid #90ee90; color: #006400; }
        .status-info.turn-black { background-color: #f0f0f0; border: 1px solid #cccccc; color: #333; }
        .status-info.thinking { background-color: #fffacd; border: 1px solid #ffd700; color: #b8860b; }
        .status-info.game-over { background-color: #add8e6; border: 1px solid #87ceeb; color: #00008b; }
        .status-info.error { background-color: #ffe4e1; border: 1px solid #ffb6c1; color: #dc143c; }
        .controls { margin-bottom: 15px; }
        .button { padding: 10px 15px; font-size: 1em; border: none; border-radius: 4px; cursor: pointer; margin: 0 5px; }
        .button-primary { background-color: #4CAF50; color: white; }
        .button-secondary { background-color: #008CBA; color: white; }
        .history-log { max-height: 200px; overflow-y: auto; border: 1px solid #ccc; padding: 10px; width: 90%; margin-top: 10px; background-color: #f9f9f9; border-radius: 4px; }
        .history-log h2 { margin: 0 0 10px 0; font-size: 1.2em; text-align: center; border-bottom: 1px solid #eee; padding-bottom: 5px; }
        #history-list { list-style: none; padding: 0; margin: 0; font-size: 0.9em; }
        #history-list li { padding: 2px 0; border-bottom: 1px dotted #eee; }
        #history-list li:last-child { border-bottom: none; }
        #history-list span { display: inline-block; width: 45%; padding: 0 2%; }
        #history-list span:first-of-type { text-align: right; }
        #history-list span:last-of-type { text-align: left; }

        /* --- NEW: Styles for the Dialog Box --- */
        .dialog-overlay { position: fixed; top: 0; left: 0; width: 100%; height: 100%; background-color: rgba(0, 0, 0, 0.6); display: none; /* Hidden by default */ justify-content: center; align-items: center; z-index: 1000; }
        .dialog-box { background-color: white; padding: 25px; border-radius: 8px; box-shadow: 0 5px 15px rgba(0,0,0,0.3); text-align: center; max-width: 400px; width: 90%; position: relative; }
        .dialog-box h3 { margin-top: 0; margin-bottom: 15px; font-size: 1.2em; color: #333; }
        .dialog-box textarea { width: 95%; min-height: 60px; margin-bottom: 15px; padding: 8px; border: 1px solid #ccc; border-radius: 4px; font-size: 1em; resize: vertical; }
        .dialog-box .dialog-buttons { display: flex; justify-content: space-around; }
        .dialog-response { margin-top: 15px; padding: 10px; background-color: #eef; border: 1px solid #ccd; border-radius: 4px; text-align: left; font-style: italic; max-height: 150px; overflow-y: auto; }
        .dialog-response.thinking { color: #888; }
        .dialog-response.error { color: #dc143c; font-weight: bold; }
        .dialog-close-btn { position: absolute; top: 10px; right: 10px; background: none; border: none; font-size: 1.5em; cursor: pointer; color: #888; }
    </style>
</head>
<body>
    <div class="container">
        <h1>Chess vs Gemini</h1>
        <p>You play as White. Drag pieces to move. Right-click your pieces to talk to them!</p>
        <!-- Container for the board -->
        <div id="myBoard" style="width: 400px"></div>
        <div id="status" class="status-info">Loading game...</div>
        <div class="controls">
            <button id="reset-game" class="button button-secondary">Reset Game</button>
        </div>
         <div id="history" class="history-log">
            <h2>Game History</h2>
            <ol id="history-list"><li>No moves yet.</li></ol>
        </div>
    </div>

    <!-- NEW: Dialog Box HTML (Initially Hidden) -->
    <div id="piece-dialog-overlay" class="dialog-overlay">
        <div class="dialog-box">
            <button id="dialog-close" class="dialog-close-btn">×</button>
            <h3 id="dialog-title">Ask the Piece</h3>
            <textarea id="dialog-question" placeholder="What should I be worried about? What are my best moves?"></textarea>
            <div id="dialog-response-area" class="dialog-response" style="display: none;"></div>
            <div class="dialog-buttons">
                <button id="dialog-submit" class="button button-primary">Ask</button>
                <button id="dialog-cancel" class="button button-secondary">Cancel</button>
            </div>
        </div>
    </div>

       <script>
        // Use jQuery's document ready for simplicity with this library
        $(document).ready(function() {
            console.log("JS: Document ready.");

            // 1. Variable Declarations
            var board = null;
            var currentFen = 'start';
            var currentGameHistory = [];
            var isPlayerTurn = true;
            var isGameOver = false;
            var isProcessingMove = false; // Flag to prevent interaction during processing
            var $status = $('#status');
            var $historyList = $('#history-list');
            var $boardElement = $('#myBoard');
            var $dialogOverlay = $('#piece-dialog-overlay');
            var $dialogTitle = $('#dialog-title');
            var $dialogQuestion = $('#dialog-question');
            var $dialogResponseArea = $('#dialog-response-area');
            var $dialogSubmit = $('#dialog-submit');
            var $dialogCancel = $('#dialog-cancel');
            var $dialogClose = $('#dialog-close');
            var currentDialogData = null;
            var isRightMouseButtonDown = false; // Flag for right mouse button detection

            const API_BASE_URL = "/api";

            // 2. Helper Function Definitions
            async function fetchApi(endpoint, method = 'GET', body = null) {
                const url = `${API_BASE_URL}${endpoint}`;
                const options = { method: method, headers: { 'Content-Type': 'application/json' } };
                if (body) { options.body = JSON.stringify(body); }
                console.log(`JS: Fetching ${method} ${url}`, body);
                try {
                    const response = await fetch(url, options);
                    const responseData = await response.json();
                    if (!response.ok) {
                         const errorMessage = responseData.error || `HTTP error! status: ${response.status}`;
                         throw new Error(errorMessage);
                    }
                    return responseData;
                } catch (error) {
                    console.error(`API Error (${method} ${url}):`, error);
                    updateStatus(`Error: ${error.message || 'Network or server error'}`, true); // Update status on fetch error
                    throw error; // Re-throw so calling function knows it failed
                }
            }

            function updateStatus(message, isError = false, isThinking = false, baseClass = null) {
                if(!$status.length) { console.error("Status element not found!"); return; }
                console.log(`JS: updateStatus - ${message}`);
                $status.text(message);
                $status.removeClass('error thinking turn-white turn-black game-over'); // Clear previous states
                if(baseClass) $status.addClass(baseClass); // Add base class (turn-white, game-over, etc.)
                if(isError) $status.addClass('error');
                if(isThinking) $status.addClass('thinking');
            }

            function updateHistoryList(history) {
                 if(!$historyList.length) return;
                 $historyList.empty();
                 currentGameHistory = history || []; // Update local copy
                 if (!currentGameHistory || currentGameHistory.length === 0) {
                     $historyList.append('<li>No moves yet.</li>'); return;
                 }
                 for (let i = 0; i < currentGameHistory.length; i += 2) {
                     const moveNumber = Math.floor(i / 2) + 1;
                     const whiteMove = currentGameHistory[i];
                     const blackMove = currentGameHistory[i + 1] || '...';
                     const listItem = `<li>${moveNumber}. <span>${whiteMove}</span> <span>${blackMove}</span></li>`;
                     $historyList.append(listItem);
                 }
                 $historyList.scrollTop($historyList[0].scrollHeight); // Scroll to bottom
            }

             function getPieceName(pieceCode) {
                const map = { 'P': 'Pawn', 'N': 'Knight', 'B': 'Bishop', 'R': 'Rook', 'Q': 'Queen', 'K': 'King' };
                // Ensure pieceCode is valid before accessing index 1
                if (pieceCode && pieceCode.length === 2) {
                    return map[pieceCode[1].toUpperCase()] || 'Unknown Piece';
                }
                return 'Unknown Piece';
            }


            // 3. State Update Function
            function updateGameState(data) {
                console.log("JS: updateGameState called with data:", data);
                if (data && data.fen) {
                    console.log("JS: updateGameState - Data is valid, processing state.");
                    const wasLoading = $('#status').text() === "Loading game..."; // Check if it's the initial load

                    currentFen = data.fen;
                    isGameOver = data.is_game_over || false;
                    isPlayerTurn = (data.turn === 'white') && !isGameOver;

                    // Determine status message and class
                    let statusMsg = "Status Error"; let statusClass = "error";
                    if (isGameOver) {
                        statusClass = "game-over";
                        if (data.winner === 'white') statusMsg = `🎉 Game Over! You Won! (${data.termination || ''})`;
                        else if (data.winner === 'black') statusMsg = `🤖 Game Over! LLM Won! (${data.termination || ''})`;
                        else statusMsg = `🤝 Game Over! Draw! (${data.termination || ''})`;
                    } else if (isPlayerTurn) {
                        statusClass = "turn-white"; statusMsg = "⚪ Your Turn (White)";
                    } else {
                        statusClass = "turn-black"; statusMsg = "⚫ LLM's Turn (Black)";
                    }
                    console.log(`JS: updateGameState - Status determined: ${statusMsg}`);

                    // Apply status update (with potential delay for initial load)
                    const updateFunc = () => {
                        updateStatus(statusMsg, false, false, statusClass); // Calls the helper
                    };

                    if (wasLoading) {
                        console.log("JS: updateGameState - Initial load detected, delaying status update slightly.");
                        setTimeout(updateFunc, 100); // Delay first update
                    } else {
                        updateFunc(); // Update immediately for subsequent calls
                    }

                    console.log("JS: updateGameState - About to update history list.");
                    updateHistoryList(data.history || []); // Calls the helper
                    console.log("JS: updateGameState - History list updated.");

                    if (board) {
                        console.log("JS: updateGameState - Board object exists, setting position and orientation.");
                        board.position(currentFen, false);
                        board.orientation('white');
                        console.log("JS: updateGameState - Board position and orientation set.");
                    } else {
                        console.log("JS: updateGameState - WARNING: Board object doesn't exist for update.");
                    }

                } else {
                     console.error("JS: updateGameState received invalid data or missing FEN:", data);
                     updateStatus("Error: Invalid game state received from server.", true); // Show error status
                     isPlayerTurn = false; // Prevent moves if state is broken
                }
                console.log("JS: updateGameState finished processing.");
            }


            // 4. Event Handler Definitions
            function onBoardMouseDown(event) {
                // event.which: 1 = left, 2 = middle, 3 = right
                if (event.which === 3) {
                    console.log("JS: Right mouse button DOWN detected.");
                    isRightMouseButtonDown = true;
                } else {
                    // Reset flag if left/middle button pressed, otherwise keep current state
                    if (event.which === 1 || event.which === 2) {
                        isRightMouseButtonDown = false;
                    }
                }
                // Allow event to proceed for Chessboard.js drag start
            }

            function onDragStart (source, piece, position, orientation) {
                console.log(`JS: onDragStart check - Source: ${source}, Piece: ${piece}, isPlayerTurn: ${isPlayerTurn}, isGameOver: ${isGameOver}, isProcessingMove: ${isProcessingMove}, isRightMBDown: ${isRightMouseButtonDown}`);

                // Prevent NORMAL drag if game over, not player's turn, black piece OR if processing move
                // Do NOT prevent if it's a right mouse down, as we need onDrop to fire for the dialog
                if (!isRightMouseButtonDown && (isGameOver || !isPlayerTurn || piece.search(/^b/) !== -1 || isProcessingMove)) {
                    if(isProcessingMove) console.log("JS: Drag prevented: Processing move.");
                    else console.log("JS: Drag prevented: State condition not met.");
                    return false; // Prevent drag
                }
                console.log("JS: Drag sequence allowed (might be left drag or right click).");
                return true; // Allow drag sequence for valid moves AND right-clicks
            }

            async function onDrop (source, target, piece, newPos, oldPos, orientation) {
                console.log(`***** JS: onDrop CALLED ***** - Source: ${source}, Target: ${target}, Piece: ${piece}, isRightMBDown: ${isRightMouseButtonDown}`);

                const rightClickDetected = isRightMouseButtonDown && source === target;
                const leftClickDetected = !isRightMouseButtonDown && source === target;

                // --- Action 1: Handle Right Click ---
                if (rightClickDetected) {
                    console.log(`JS: Right-click detected in onDrop for piece ${piece} on ${source}.`);
                    isRightMouseButtonDown = false; // Reset flag immediately

                    // Perform checks needed before opening dialog
                    if (isGameOver || !isPlayerTurn || isProcessingMove) {
                        console.log("JS: Ignoring right-click action: State prevents it."); return;
                    }
                    if (!piece || piece.search(/^w/) === -1) {
                         console.log("JS: Ignoring right-click action: Not a valid white piece."); return;
                    }

                    // --- Open the Dialog ---
                    console.log(`JS: Opening dialog for right-clicked piece: ${piece} on ${source}`);
                    openPieceDialog(piece, source); // Open dialog

                    return; // IMPORTANT: Stop processing, don't treat as a move
                }

                // --- Action 2: Handle Left Click (No Drag) ---
                // Reset right mouse flag if it wasn't already handled
                isRightMouseButtonDown = false;
                if (leftClickDetected) {
                    console.warn("JS: onDrop ignored - source and target are the same (non-right click).");
                    return 'snapback'; // Snap back visually for left-click
                }

                // --- Action 3: Handle Actual Move (Drag Drop) ---
                // State checks (redundant with onDragStart but good defense)
                if (isGameOver || !isPlayerTurn || isProcessingMove) {
                     console.warn(`JS: onDrop ignored - State prevents move (Turn: ${isPlayerTurn}, Over: ${isGameOver}, Processing: ${isProcessingMove})`);
                     return 'snapback';
                 }

                // Proceed with normal move logic
                console.log('JS: onDrop proceeding with normal move calculation.');
                var moveUCI = source + target;
                // Handle promotion
                if (piece === 'wP' && target.charAt(1) === '8') { moveUCI += 'q'; console.log("Auto-promoting to Queen"); }
                if (piece === 'bP' && target.charAt(1) === '1') { moveUCI += 'q'; } // Defensive

                isProcessingMove = true; // Block other actions
                updateStatus("⚪ Sending move...", false, true, "turn-white thinking"); // Update status

                try {
                    const data = await fetchApi('/move', 'POST', { move: moveUCI }); // Send move to backend
                    console.log("JS: Move API response received in onDrop:", data);
                    updateGameState(data); // Update UI with result (includes black's move)
                } catch (error) {
                    console.error("JS: Move failed in onDrop:", error.message);
                    // Status updated by fetchApi catch block
                    if (board) board.position(currentFen); // Revert visual board on error
                    return 'snapback'; // Animate piece snapback
                } finally {
                     isProcessingMove = false; // Allow actions again
                     console.log("JS: Move processing finished in onDrop. isProcessingMove set to false.");
                }
                // Don't return 'snapback' on success
            }


            // 5. Dialog Logic Functions
            function openPieceDialog(piece, square) {
                currentDialogData = { piece, square };
                const pieceName = getPieceName(piece); // Calls helper

                $dialogTitle.text(`Ask the ${pieceName} on ${square}`);
                $dialogQuestion.val('');
                $dialogResponseArea.hide().empty().removeClass('error thinking');
                $dialogOverlay.css('display', 'flex'); // Show dialog
                $dialogQuestion.focus();
                console.log("JS: Piece Dialog opened.");
            }

            function closePieceDialog() {
                $dialogOverlay.hide();
                currentDialogData = null;
                console.log("JS: Piece Dialog closed.");
            }

            async function submitPieceQuestion() {
                if (!currentDialogData) return;
                const question = $dialogQuestion.val().trim();
                if (!question) { alert("Please enter a question for the piece."); return; }

                const { piece, square } = currentDialogData;
                $dialogResponseArea.text('Thinking...').addClass('thinking').removeClass('error').show(); // Show thinking state
                $dialogSubmit.prop('disabled', true); // Disable button

                try {
                    const requestBody = {
                        question: question, piece: piece, square: square,
                        fen: currentFen, history: currentGameHistory // Send current state
                    };
                    const data = await fetchApi('/ask_piece', 'POST', requestBody); // Call backend
                    if (data && data.answer) {
                        $dialogResponseArea.text(data.answer).removeClass('thinking error'); // Display answer
                    } else {
                         $dialogResponseArea.text('Received an unexpected response.').addClass('error').removeClass('thinking');
                    }
                } catch (error) {
                     // Error message potentially set by fetchApi status update
                     $dialogResponseArea.text(`Error: ${error.message || 'Could not get answer.'}`).addClass('error').removeClass('thinking');
                } finally {
                     $dialogSubmit.prop('disabled', false); // Re-enable button
                }
            }


            // 6. Initialization Function
            async function initializeGame() {
                console.log("JS: initializeGame() starting.");
                updateStatus("Loading game...", false, true); // Uses updateStatus
                isProcessingMove = true; // Block interaction during load
                console.log("JS: Set isProcessingMove = true.");

                try {
                    console.log("JS: About to call fetchApi('/game')...");
                    const data = await fetchApi('/game'); // Uses fetchApi
                    console.log("JS: fetchApi('/game') returned. Data received:", data);

                    if (!data || typeof data.fen !== 'string') {
                         console.error("JS: ERROR - Invalid or missing data/fen from /api/game. Data:", data);
                         throw new Error("Missing or invalid FEN in initial game data.");
                    }
                    console.log("JS: Data seems valid. FEN:", data.fen);

                    currentFen = data.fen;
                    currentGameHistory = data.history || [];
                    console.log("JS: Set currentFen and currentGameHistory.");

                    var config = {
                        draggable: true,
                        position: currentFen,
                        orientation: 'white',
                        onDragStart: onDragStart,
                        onDrop: onDrop,
                        pieceTheme: '/static/{piece}.png',
                        // --- Adjust Speeds Here ---
                        moveSpeed: 500,         // Faster animation (milliseconds, lower is faster)
                        snapbackSpeed: 500,   // Faster snapback on illegal move (default 500)
                        snapSpeed: 100       // Faster snap to square on drop (default 100)
                        // --- End of Speed Adjustments ---
                    };
                    console.log("JS: Config created. About to initialize Chessboard...");

                    if (!$('#myBoard').length) {
                        console.error("JS: ERROR - #myBoard element not found in DOM!");
                        throw new Error("Chessboard container element #myBoard not found.");
                    }

                    board = Chessboard('myBoard', config); // Initialize chessboard UI
                    console.log("JS: Chessboard object potentially created.");

                    if (!board || typeof board.position !== 'function') {
                         console.error("JS: ERROR - Chessboard initialization failed! Board object invalid or null.", board);
                         throw new Error("Failed to initialize chessboard UI library.");
                    }
                    console.log("JS: Chessboard object confirmed created and seems valid.");

                    // Bind mouse down AFTER board is initialized
                    console.log("JS: About to bind mousedown listener...");
                    $boardElement.on('mousedown', onBoardMouseDown); // Must be defined above
                    console.log("JS: Mousedown listener bound.");

                    console.log("JS: About to call updateGameState...");
                    updateGameState(data); // Update UI based on fetched state (uses updateGameState)
                    console.log("JS: Initial updateGameState finished.");

                } catch (error) {
                    console.error("JS: Initialization failed inside CATCH block:", error);
                    // Attempt to update status even on init failure
                    try { updateStatus(`Initialization failed: ${error.message || 'Unknown error'}`, true); }
                    catch(statusError) { console.error("JS: CRITICAL - Failed to update status within init catch block:", statusError); }
                } finally {
                    isProcessingMove = false; // Allow interaction once init is done (or failed)
                    console.log("JS: initializeGame finally block reached. isProcessingMove set to false.");
                }
            }


            // 7. Event Listener Bindings (Buttons, Dialog)
             $('#reset-game').on('click', async function() {
                 console.log("JS: Reset button clicked.");
                 if (isProcessingMove) return; // Prevent multiple resets
                 isProcessingMove = true; // Prevent interaction during reset
                 closePieceDialog(); // Close dialog if open
                 updateStatus("🔄 Resetting game...", false, true);
                 try {
                    const data = await fetchApi('/reset', 'POST');
                    console.log("Reset API response:", data);
                    updateGameState(data); // Update UI to reset state
                 } catch (error) {
                     console.error("Reset failed.");
                     // Status already updated by fetchApi catch block
                 } finally {
                     isProcessingMove = false; // Re-enable interaction
                 }
             });

            $dialogSubmit.on('click', submitPieceQuestion); // Uses submitPieceQuestion
            $dialogCancel.on('click', closePieceDialog);    // Uses closePieceDialog
            $dialogClose.on('click', closePieceDialog);     // Uses closePieceDialog
            $dialogQuestion.on('keypress', function(e) {
                if (e.which === 13 && !e.shiftKey) { // Enter key without shift
                    e.preventDefault(); // Prevent newline in textarea
                    submitPieceQuestion(); // Trigger submit
                }
            });


            // 8. Start Application
            initializeGame(); // Call the initialization function

        }); // End of $(document).ready
    </script>
</body>
</html>
    """
    return Response(html_content, mimetype='text/html')

# --- Main Execution ---
if __name__ == '__main__':
    print("Starting Flask server (Talk to Pieces Feature Enabled)...")
    print(f"API Key Loaded: {'Yes' if GEMINI_API_KEY else 'No'}")
    print(f"Gemini Model: {MODEL_NAME if genai_model else 'Not Initialized'}")
    # Set debug=False for production deployment
    # Use reloader can sometimes cause issues with global state, but fine for dev
    app.run(host='127.0.0.1', port=5000, debug=True, use_reloader=True)
