
#include "cuda_runtime.h"
#include "device_launch_parameters.h"

#include <stdio.h>
#include <stdlib.h>
#include <string.h>

#define DEFAULT_SIZE 5

typedef struct neuron_T Neuron;
typedef struct neuron_T
{
	float value;
	float* weights;
	Neuron* connected_to;
} Neuron;

typedef struct layer_T
{
	Neuron* neurons;
} Layer;

void main_function(const char* content)
{
	Layer starting_layer;
	Layer last_layer;

	// ----------------------------------------------------

	starting_layer.neurons = (Neuron*)malloc(sizeof(Neuron) * 4);
	last_layer.neurons = (Neuron*)malloc(sizeof(Neuron) * 2);

	last_layer.neurons[0].value = 0.0;
	last_layer.neurons[1].value = 0.0;

	for (size_t i = 0; i < strlen(content); i++)
	{
		float weight1 = (i == 0 || i == 3) ? 0.5 : -0.5;
		float weight2 = (i == 0 || i == 3) ? -0.5 : 0.5;

		Neuron neuron = { 0 };
		neuron.value = float(content[i]);

		neuron.weights = (float*)malloc(sizeof(float) * 2);
		neuron.weights[0] = weight1; neuron.weights[1] = weight2;

		neuron.connected_to = (Neuron*)malloc(sizeof(Neuron) * 2);
		neuron.connected_to[0] = last_layer.neurons[0];  neuron.connected_to[1] = last_layer.neurons[1];

		starting_layer.neurons[i] = neuron;
	}

	// ----------------------------------------------------

	for (size_t i = 0; i < 4; i++)
	{
		Neuron neuron = starting_layer.neurons[i];

		last_layer.neurons[0].value += neuron.value * neuron.weights[0];
		last_layer.neurons[1].value += neuron.value * neuron.weights[1];
	}

	// ----------------------------------------------------

	if (last_layer.neurons[0].value == 1)
	{
		printf("Left white diagonal.\n");
	}

	else if (last_layer.neurons[1].value == 1)
	{
		printf("Right white diagonal.\n");
	}

	else
	{
		printf("No diagonal.\n");
	}

	printf("Neuron 0: %.2f\n", last_layer.neurons[0].value);
	printf("Neuron 0: %.2f\n", last_layer.neurons[1].value);

	// ----------------------------------------------------

	for (size_t i = 0; i < 4; i++)
	{
		free(starting_layer.neurons[i].weights);
		free(starting_layer.neurons[i].connected_to);
	}

	free(starting_layer.neurons);
	free(last_layer.neurons);
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
				char* temp = (char*)malloc(sizeof(char) * DEFAULT_SIZE);
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
			}

			content[i] = c;
		}
	}

	content[i + 1] = '\0';

	main_function(content);
	free(content);
}
