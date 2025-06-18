
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
		multiply_layer(network[i], network[i + 1]);
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
