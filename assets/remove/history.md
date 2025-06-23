
1. minimax + alpha beta: mô hình chạy ổn định, có nhiều phương pháp cải thiện mô hình như: cải thiện tìm kiếm, tìm kiếm ưu tiên, hiệu quả nhất (có ảnh hưởng lớn tới hiệu quả mô hình) là cải thiện khả năng search và evaluation mô hình(ảnh hưởng cực lớn, cần phải có hiểu biết sâu)

2. neural network: có nhiều loại mô hình tuỳ vào dữ liệu đầu vào: fen, pgn + các kiến trúc khá basic(hiệu quả cao) tuy nhiên cần rất nhiều data

3. rl


nnue: chạy được, tuy nhiên việc đưa đầu vào không phân biệt white, black -> mô hình mising context 
Input: 64 squares * 13 piece types: idea cơ bản, mô hình có hoạt động

nnue_2: xây dựng khá khó hiểu đoạn input đầu vào
fw_tensor, offsets_white, fb_tensor, offsets_black, stm_tensor, target_tensor: lấy offsets trước của cả trắng, đen, tuy nhiên phải setup chạy full 1 match thì mới đúng, dữ liệu fen hiện tại là đầu vào riêng lẻ