from processing_code import input_data, import_data, split_data, train_model, prediction, evaluation_model, relation_map, rf_importance_graph, compare_pred_real

input_file_name = input_data()
data = import_data(input_file_name)
A = split_data(data)
print(A[4])
B = train_model(A[0],A[1],A[2],A[3])
C = prediction(B[0],B[1],B[2],B[3],A[1],A[4])
D = evaluation_model(A[3], C[0], C[1], C[2], C[3])
col = relation_map(data, A[4])
rf_importance_graph(col, B[0], B[1], B[2], B[3])
compare_pred_real(A[3], C[0], C[1], C[2], C[3])