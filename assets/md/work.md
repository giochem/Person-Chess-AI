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
    - model_architecture: structure model (each neural have neural network in .ipynb and full ChessAi in .py)
    - *.ipynb: experiment for chess ai(train and test one example to know setup run neutral network)
    - test.ipynb: apply test case check model run correct for spec case
    - compare.ipynb: just check current model is the best

PLAN: train model -> gắn vào basic minimax thay eval ban đầu -> compare 
        -> thay đổi mô hình nếu thấy vẫn ko tốt (nnue)
        -> build nnue_3 -> tốt thì lấy, ko thì lấy từ chess-engine
        -> chuyển sang xây dựng ứng dụng: fastapi, web socket


        - còn time: tìm evaluation tối ưu -> compare
                     apply rl for this -> compare
                     danh sách các mô hình + source code, có so sánh và thi đấu
                     mang lên lichess thi đấu
                     tối ưu hoá: chuyển sang ngôn ngữ chạy nhanh hơn: c++, go, java
                                 thuật toán tối ưu chạy train + code: gpu, bitboard,
