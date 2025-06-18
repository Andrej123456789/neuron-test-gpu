
#include "cuda_runtime.h"
#include "device_launch_parameters.h"

#include <ctype.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

#define DEFAULT_SIZE 5
#define LAYERS_IN_NETWORK 2

typedef struct neuron_T
{
	float value;
	float* weights;
	size_t weights_len;
} Neuron;

typedef struct layer_T
{
	Neuron* neurons;
	size_t neurons_len;
} Layer;

__global__ void multiply_layer_kernel(float* input_values, float* input_weights, size_t input_neurons_len,
	float* output_values, size_t output_neurons_len)
{
	int j = blockIdx.x * blockDim.x + threadIdx.x;
	if (j < output_neurons_len)
	{
		for (size_t i = 0; i < input_neurons_len; ++i)
		{
			output_values[j] += input_values[i] * input_weights[i * output_neurons_len + j];
		}
	}
}

Layer* initialize_layer(int size)
{
	Layer* layer = (Layer*)malloc(sizeof(Layer));
	if (layer == NULL)
	{
		printf("Malloc error!\n");
		return NULL;
	}

	layer->neurons = (Neuron*)calloc(size, sizeof(Neuron));
	if (layer->neurons == NULL)
	{
		printf("Malloc error!\n");
		return NULL;
	}

	layer->neurons_len = size;
	return layer;
}

void assign_starting_layer(Layer* layer, int* content)
{
	for (size_t i = 0; i < layer->neurons_len; i++)
	{
		float weight1 = (i == 0 || i == 3) ? 0.5 : -0.5;
		float weight2 = (i == 0 || i == 3) ? -0.5 : 0.5;

		layer->neurons[i].value = float(content[i]);

		layer->neurons[i].weights = (float*)calloc(2, sizeof(float));
		if (layer->neurons[i].weights == NULL)
		{
			printf("Malloc error!\n");

			layer->neurons[i].value = 0.0;
			return;
		}

		layer->neurons[i].weights_len = 2;

		layer->neurons[i].weights[0] = weight1;
		layer->neurons[i].weights[1] = weight2;
	}
}

void multiply_layer(Layer* first_layer, Layer* second_layer)
{
	for (size_t i = 0; i < first_layer->neurons_len; i++)
	{
		for (size_t j = 0; j < second_layer->neurons_len; j++)
		{
			second_layer->neurons[j].value += first_layer->neurons[i].value * first_layer->neurons[i].weights[j];
		}
	}
}

void multiply_layer_cuda(Layer* first_layer, Layer* second_layer)
{
	size_t input_neurons_len = first_layer->neurons_len;
	size_t output_neurons_len = second_layer->neurons_len;

	// Allocate host-side flat arrays
	float* h_input_values = (float*)malloc(sizeof(float) * input_neurons_len);
	float* h_input_weights = (float*)malloc(sizeof(float) * input_neurons_len * output_neurons_len);
	float* h_output_values = (float*)malloc(sizeof(float) * output_neurons_len);

	// Fill host arrays
	for (size_t i = 0; i < input_neurons_len; ++i)
	{
		h_input_values[i] = first_layer->neurons[i].value;
		for (size_t j = 0; j < output_neurons_len; ++j)
		{
			h_input_weights[i * output_neurons_len + j] = first_layer->neurons[i].weights[j];
		}
	}

	for (size_t j = 0; j < output_neurons_len; ++j)
	{
		h_output_values[j] = second_layer->neurons[j].value;
	}

	// Allocate device memory
	float* d_input_values, * d_input_weights, * d_output_values;
	cudaMalloc(&d_input_values, sizeof(float) * input_neurons_len);
	cudaMalloc(&d_input_weights, sizeof(float) * input_neurons_len * output_neurons_len);
	cudaMalloc(&d_output_values, sizeof(float) * output_neurons_len);

	// Copy to device
	cudaMemcpy(d_input_values, h_input_values, sizeof(float) * input_neurons_len, cudaMemcpyHostToDevice);
	cudaMemcpy(d_input_weights, h_input_weights, sizeof(float) * input_neurons_len * output_neurons_len, cudaMemcpyHostToDevice);
	cudaMemcpy(d_output_values, h_output_values, sizeof(float) * output_neurons_len, cudaMemcpyHostToDevice);

	// Launch kernel
	int threads_per_block = 256;
	int blocks_per_grid = (output_neurons_len + threads_per_block - 1) / threads_per_block;
	multiply_layer_kernel <<< blocks_per_grid, threads_per_block >> > (d_input_values, d_input_weights, input_neurons_len, d_output_values, output_neurons_len);

	// Copy results back
	cudaMemcpy(h_output_values, d_output_values, sizeof(float) * output_neurons_len, cudaMemcpyDeviceToHost);

	// Update second layer values
	for (size_t j = 0; j < output_neurons_len; ++j)
	{
		second_layer->neurons[j].value = h_output_values[j];
	}

	// Cleanup
	free(h_input_values);
	free(h_input_weights);
	free(h_output_values);

	cudaFree(d_input_values);
	cudaFree(d_input_weights);
	cudaFree(d_output_values);
}

void main_function(int* content)
{
	Layer* network[LAYERS_IN_NETWORK];

	network[0] = initialize_layer(4);
	if (network[0] == NULL)
	{
		return;
	}

	network[1] = initialize_layer(2);
	if (network[1] == NULL)
	{
		return;
	}

	assign_starting_layer(network[0], content);

	for (size_t i = 0; i < LAYERS_IN_NETWORK - 1; i++) // do not multiply last layer
	{
		multiply_layer_cuda(network[i], network[i + 1]);
	}

	if (network[1]->neurons[0].value == 1)
	{
		printf("Left white diagonal.\n");
	}

	else if (network[1]->neurons[1].value == 1)
	{
		printf("Right white diagonal.\n");
	}

	else
	{
		printf("No diagonal.\n");
	}

	printf("Neuron 0: %.2f\n", network[1]->neurons[0].value);
	printf("Neuron 0: %.2f\n", network[1]->neurons[1].value);

	for (size_t i = 0; i < network[0]->neurons_len; i++)
	{
		free(network[0]->neurons[i].weights);
	}

	free(network[0]->neurons);
	free(network[1]->neurons);

	free(network[0]);
	free(network[1]);
}

int main(int argc, char* argv[])
{
	char path[256];
	int content[4];

	printf("--------------------\n");
	printf("GPU neuron network test\n");
	printf("--------------------\n");

	if (argc < 2)
	{
		printf("Enter a path to the file: ");
		scanf("%255s", path);
	}

	else
	{
		strncpy(path, argv[1], 255);
		path[255] = '\0';
	}

	FILE* file;
	file = fopen(path, "r");

	if (file == NULL)
	{
		printf("File not found!\n");
		printf("Exiting ...\n");

		return 0;
	}

	char c;
	int i = 0;

	while ((c = fgetc(file)) != EOF)
	{
		if (isdigit(c))
		{
			content[i++] = c - '0';
		}
	}

	main_function(content);
}
