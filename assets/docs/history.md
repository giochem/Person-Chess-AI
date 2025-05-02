# Chess AI Documentation

## Model Approaches

### 1. Minimax + Alpha Beta

- Model runs stably
- Improvement methods:
  - Search optimization
  - Priority search
  - Search capability improvement (major impact)
  - Evaluation model enhancement (requires deep understanding)

### 2. Neural Network

- Various models depending on input data: FEN, PGN
- Basic architectures with high effectiveness
- Requires substantial training data

### 3. NNUE Models

#### NNUE Version 1

- Operational but lacks white/black context distinction
- Input: 64 squares \* 13 piece types
- Basic idea with functional model

#### NNUE Version 2

- Complex input structure
- Components: fw_tensor, offsets_white, fb_tensor, offsets_black, stm_tensor, target_tensor
- Requires full match setup for correct operation

## Implementation Notes

### Development Guidelines

- During preprocessing: ensure consistent data types (tensor, numpy, array)
- Verify data dimensions throughout the model pipeline (pre, during, post training)

### Algorithm Components

- Minimax
- Alpha-Beta pruning
- Piece-Square Table
- Iterative Deepening + Move ordering

### Evaluation Factors

- Piece activity
- King safety
- Pawn structure
- Piece-Square Tables (opening, middle, end)
- Endgame Tablebases

## Version Development

### V1

- Setup model (fast and easy to implement)

### V2

- Deeper implementation with neural network

### V3

- Comparison
- Hard FEN testing
- Web demo
- Documentation

## Project Structure

### Notebook Organization

- checkpoint/: Neural network model saves
- model_architecture/: Model structure definitions
- \*.ipynb: Chess AI experiments
- test.ipynb: Specific test cases
- compare.ipynb: Model comparison

## Development Plan

1. Train model
2. Integrate with basic minimax replacing initial eval
3. Model comparison
4. Switch to NNUE if needed
5. Build NNUE_3 or use chess-engine
6. Application development:
   - FastAPI
   - WebSocket

### Future Improvements

- Optimize evaluation
- Apply reinforcement learning
- Document model comparisons
- Lichess competition
- Performance optimization:
  - Language migration (C++, Go, Java)
  - Algorithm optimization
  - Training optimization (GPU, bitboard)

## Current Focus

- Implementing main evaluation
- Reference: [Stockfish Evaluation Guide](https://hxim.github.io/Stockfish-Evaluation-Guide/)
- Creating JavaScript-based minimax + evaluation
- Chess engine development for Vue.js integration
