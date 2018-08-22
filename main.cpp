
#include <string>
#include <iostream>

//using namespace std;
//
//int main(int argc, char* argv[]) {
//	string output_path = "D:\\";
//	for (size_t i = 0; i < argc; i++)
//	{
//		if (strcmp(argv[i], "--output_path") == 0)
//		{
//			i++;
//			output_path = argv[i];
//		}
//	}
//	cout << output_path << endl;
//	system("pause");
//	return 0;
//}

#include<tensorflow\core\public\session.h>
#include "tensorflow/core/protobuf/meta_graph.pb.h"
//#include "tensorflow/cc/client/client_session.h"
//#include "tensorflow/cc/ops/standard_ops.h"
//#include "tensorflow/core/framework/tensor.h"

using namespace tensorflow;
using namespace std;

int main(int argc, char* argv[])
{
	string input_path = "D:\\ActionBasedFX\\origin_output\\";
	for (size_t i = 0; i < argc; i++)
	{
		if (strcmp(argv[i], "--output_path") == 0)
		{
			i++;
			input_path = argv[i];
		}
	}

	const string pathToGraph = input_path + "model_950.ckpt.meta";
	const string checkpointPath = input_path + "model_950.ckpt";
	auto session = NewSession(SessionOptions());
	if (session == nullptr)
	{
		throw runtime_error("Could not create Tensorflow session.");
	}

	Status status;

	// 读入我们预先定义好的模型的计算图的拓扑结构
	MetaGraphDef graph_def;
	status = ReadBinaryProto(Env::Default(), pathToGraph, &graph_def);
	if (!status.ok())
	{
		throw runtime_error("Error reading graph definition from " + pathToGraph + ": " + status.ToString());
	}

	// 利用读入的模型的图的拓扑结构构建一个session
	status = session->Create(graph_def.graph_def());
	if (!status.ok())
	{
		throw runtime_error("Error creating graph: " + status.ToString());
	}

	// 读入预先训练好的模型的权重
	Tensor checkpointPathTensor(DT_STRING, TensorShape());
	checkpointPathTensor.scalar<std::string>()() = checkpointPath;
	status = session->Run(
		{ { graph_def.saver_def().filename_tensor_name(), checkpointPathTensor }, },
		{},
		{ graph_def.saver_def().restore_op_name() },
		nullptr);
	if (!status.ok())
	{
		throw runtime_error("Error loading checkpoint from " + checkpointPath + ": " + status.ToString());
	}

	vector<pair<string, Tensor>> inputs;
	// todo: TensorShape x0.tensor<float, 3>() output/y
	Tensor batch_size(tensorflow::DT_INT32, TensorShape());
	batch_size.scalar<int>()() = 1;
	
	Tensor keep_prob(tensorflow::DT_FLOAT, TensorShape());
	keep_prob.scalar<float>()() = 1.0;

	//PartialTensorShape x0_sp({ 1, 24, 36 });
	Tensor x(tensorflow::DT_FLOAT, TensorShape({ 1, 24 ,36 }));

	//Tensor label(tensorflow::DT_FLOAT, TensorShape({ 1, 4 }));

	//input data
	auto x_map = x.tensor<float, 3>();
	//auto label_map = label.tensor<float, 2>();

	for (size_t i = 0; i < 24; i++)
	{
		for (size_t j = 0; j < 36; j++)
		{
			x_map(0, i, j) = 1.0;
		}
	}
	//for (size_t i = 0; i < 4; i++)
	//{
	//	label_map(0, i) = 0.0;
	//}
	inputs.emplace_back(string("batch_size"), batch_size);
	inputs.emplace_back(string("keep_prob"), keep_prob);
	inputs.emplace_back(string("x"), x);
	//inputs.emplace_back(string("label"), label);

	//  构造模型的输入，相当与python版本中的feed
	//std::vector<std::pair<string, Tensor>> input;
	//tensorflow::TensorShape inputshape;
	//inputshape.InsertDim(0, 1);
	//Tensor a(tensorflow::DT_INT32, inputshape);
	//Tensor b(tensorflow::DT_INT32, inputshape);
	//auto a_map = a.tensor<int, 1>();
	//a_map(0) = 2;
	//auto b_map = b.tensor<int, 1>();
	//b_map(0) = 3;
	//input.emplace_back(std::string("a"), a);
	//input.emplace_back(std::string("b"), b);

	//   运行模型，并获取输出
	std::vector<tensorflow::Tensor> answer;
	status = session->Run(inputs, { "y" }, {}, &answer);

	Tensor y = answer[0];
	auto result_map = y.tensor<int, 1>();
	cout << "result: " << result_map(0) << endl;

	return 0;
}