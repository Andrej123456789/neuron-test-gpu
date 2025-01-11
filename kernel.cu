
#include "cuda_runtime.h"
#include "device_launch_parameters.h"

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

__global__ void multiplyLayers(Layer* first_layer, Layer* second_layer)
{
	int i = threadIdx.x;
	int j = threadIdx.y;

	second_layer->neurons[j].value += first_layer->neurons[i].value * first_layer->neurons[i].weights[j];
	return;
}

Layer* initialize_layer(int size)
{
	Layer* layer = (Layer*)malloc(sizeof(Layer));
	if (layer == NULL)
	{
		printf("Malloc error!\n");
		return NULL;
	}

	layer->neurons = (Neuron*)malloc(sizeof(Neuron) * size);
	if (layer->neurons == NULL)
	{
		printf("Malloc error!\n");
		return NULL;
	}

	layer->neurons_len = size;
	for (int i = 0; i < size; i++)
	{
		layer->neurons[i].value = 0.0;
		layer->neurons[i].weights = NULL;
		layer->neurons[i].weights_len = 0;
	}

	return layer;
}

void assign_starting_layer(Layer* layer, const char* content)
{
	for (size_t i = 0; i < layer->neurons_len; i++)
	{
		float weight1 = (i == 0 || i == 3) ? 0.5 : -0.5;
		float weight2 = (i == 0 || i == 3) ? -0.5 : 0.5;

		layer->neurons[i].value = float(content[i]);

		layer->neurons[i].weights = (float*)malloc(sizeof(float) * 2);
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
	Layer* cuda_first_layer = { 0 };
	Layer* cuda_second_layer = { 0 };

	cudaMalloc(&cuda_first_layer, sizeof(Layer));
	cudaMalloc(&cuda_second_layer, sizeof(Layer));

	cudaMemcpy(cuda_first_layer, first_layer, sizeof(Layer), cudaMemcpyHostToDevice);
	cudaMemcpy(cuda_second_layer, second_layer, sizeof(Layer), cudaMemcpyHostToDevice);

	multiplyLayers <<< 1, first_layer->neurons_len, second_layer->neurons_len >>> (cuda_first_layer, cuda_second_layer);
	cudaMemcpy(second_layer, cuda_second_layer, sizeof(Layer), cudaMemcpyDeviceToHost);

	cudaFree(cuda_first_layer);
	cudaFree(cuda_second_layer);
}

void main_function(const char* content)
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
		// multiply_layer(network[i], network[i + 1]);
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

	free(network[0]->neurons);
	free(network[1]->neurons);

	free(network[0]);
	free(network[1]);
}

int main(int argc, char* argv[])
{
	char path[256];
	char* content = (char*)malloc(sizeof(char) * DEFAULT_SIZE);

	if (content == NULL)
	{
		printf("Malloc error!\n");
		return 0;
	}

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
		path[256] = '\0';
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
	int i = -1;

	while ((c = fgetc(file)) != EOF)
	{
		if (c != ' ' && c != '\n')
		{
			i++;
			if (i >= DEFAULT_SIZE)
			{
				char* temp = (char*)malloc(sizeof(char) * (i - 1) + 1);
				if (temp == NULL)
				{
					printf("Malloc error!\n");
					exit(0);
				}

				strcpy(temp, content);

				content = (char*)realloc(content, sizeof(char) * i + 1); // +1 for \0 below
				if (content == NULL)
				{
					printf("Malloc error!\n");
					exit(0);
				}

				strcpy(content, temp);
				free(temp);
			}

			content[i] = c;
		}
	}

	content[i + 1] = '\0';

	main_function(content);
	free(content);
}
