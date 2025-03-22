Algorithm
Minimax,
Alpha-Beta pruning
Piece-Square Table
Iterative Deepening + Move ordering

V1:
rl: setup model(fast and easy to implement)

improve:
    Opening Book and Endgame Databases
    Opening Book: Deep Blue had access to a vast database of chess openings, allowing it to play established opening moves effectively.
    Endgame Tablebases: For simplified endgame scenarios, Deep Blue used precomputed databases that contained perfect play for specific positions, ensuring optimal moves.

Evaluation Function
    Deep Blue used a sophisticated evaluation function that assessed the strength of a position based on various factors, such as material balance, piece activity, king safety, and pawn structure. This function was fine-tuned using expert knowledge from chess players.



V2:
rl: deep model(improve, research paper)
app: not much, just display more model