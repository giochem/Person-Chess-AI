Algorithm
Minimax,
Alpha-Beta pruning
Piece-Square Table
Iterative Deepening + Move ordering

Evaluation: piece activity, king safety, and pawn structure
Piece-Square Tables: opening, middle, end
Endgame Tablebases: book or remember

V1:
setup model(fast and easy to implement)

V2:
deeper: neural network

V3: compare, check hard FEN, demo web, document


structure code
notebook
    - checkpoint: save model nẻualnetwork after training
    - model_architecture: structure model
    - *.ipynb: experiment for chess ai
    - test.ipynb: apply test case check model run correct for spec case
    - compare.ipynb: 