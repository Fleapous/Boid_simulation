﻿// Include CUDA headers first
#include "cuda_runtime.h"
#include "device_launch_parameters.h"
#include <thrust/sort.h>
#include <thrust/device_ptr.h>
#include <thrust/device_vector.h>
#include <thrust/execution_policy.h>


// Then include OpenGL headers
#include <glad/glad.h>
#include <GLFW/glfw3.h>

//classic headers
#include <stdio.h>
#include <cstdlib>
#include <ctime>  
#include <cmath>
#include <iostream>

#define BoidCount 100
#define CELL_ARRAY_SIZE 25

#define WIDTH 500
#define HEIGHT 500

#define HIGHNUMBER 100000
#define MAX_BOIDS_IN_A_CELL 96 //since block can only have so many threads i need to clamp it at some number (multitute of 32 will be better)

//boid logic parameters
#define PROTECTED_RANGE 5
#define AVOIDFACTOR 0.2

uint2 gridList[BoidCount];

struct vec2 {
    float x;
    float y;
};
class Cell {
public:
    int Id;
    vec2 position;

    Cell() : Id(0), position({ 0.0f, 0.0f }) {}
    Cell(int id, vec2 pos) : Id(id), position(pos) {}
};
class Boid {
public:
    int Id;
    vec2 position;
    float2 velocity;

    Boid() : Id(0), position({ 0.0f, 0.0f }), velocity({ 10,10 }) {}
    Boid(int id, vec2 pos) : Id(id), position(pos), velocity({ 10,10 }) {}
};
struct CompareX {
    __host__ __device__
        bool operator()(const uint2& a, const uint2& b) const {
        return a.x < b.x;
    }
};

void initializeBoids(int width, int height, Boid(&boidArray)[BoidCount], int size);
void SortGridList(uint2* d_gridList);
void initializeCells(int width, int height, Cell(&cellArray)[CELL_ARRAY_SIZE], int cellArraySize);
void CreateLookUpTable(uint2* d_gridList, Boid* d_boidArray, Cell* d_cellArray, int* d_lookUpTable);
void CalculateBoidLogic(uint2* d_gridList, Boid* d_boidArray, Cell* d_cellArray, int* d_lookUpTable, float deltaTime);
void framebuffer_size_callback(GLFWwindow* window, int width, int height);
void processInput(GLFWwindow* window);

void checkCudaError(cudaError_t cudaStatus, const char* errorMessage) {
    if (cudaStatus != cudaSuccess) {
        std::cerr << errorMessage << " failed: " << cudaGetErrorString(cudaStatus) << "\n";
        exit(EXIT_FAILURE);
    }
}

void checkCudaLastError(const char* msg) {
    cudaError_t cuda_error = cudaGetLastError();
    if (cuda_error != cudaSuccess) {
        std::cerr << "CUDA error after " << msg << ": " << cudaGetErrorString(cuda_error) << std::endl;
        // Handle the error appropriately, e.g., exit the program
        exit(EXIT_FAILURE);
    }
}

__global__ void hashBoids(Boid* boidArray, uint2* gridList, Cell* cellArray, int boidCount, int cellArraySize, int width, int height) {
    int boidIndex = threadIdx.x + blockIdx.x * blockDim.x;

    if (boidIndex < boidCount) {
        //new boidindex finding alg
        int boidXPos = static_cast<int>(static_cast<int>(boidArray[boidIndex].position.y / (HEIGHT / sqrt((float)cellArraySize))) * sqrt((float)cellArraySize)); // only need the integer part 
        int boidYPos = static_cast<int>(static_cast<int>(boidArray[boidIndex].position.x / (WIDTH / sqrt((float)cellArraySize))));
        int cellIndex = boidXPos + boidYPos;
        gridList[boidIndex] = make_uint2(cellIndex, boidIndex);

        //printf("boidPosX: %f, boidPosY: %f, boidXPos: %d, boidYPos: %d, cellIndex: %d\n",
        //    boidArray[boidIndex].position.x,boidArray[boidIndex].position.y, boidXPos, boidYPos, cellIndex);
        //checking if cell is a corner or a side
        // temporarry 
        width = 5;
        height = 5;

        int xOffset = cellIndex % width;
        int yOffset = static_cast<int>((cellIndex - xOffset) / height);

        //adding neighbors 
        int stride = boidCount;
        //left cell
        if (xOffset != 0)
            gridList[boidIndex + stride] = make_uint2(cellIndex - 1, boidIndex);

        //right
        if (xOffset != width - 1)
            gridList[boidIndex + stride * 2] = make_uint2(cellIndex + 1, boidIndex);

        //top cell
        if (yOffset != 0)
            gridList[boidIndex + stride * 3] = make_uint2(cellIndex - width, boidIndex);


        //bottom cell
        if (yOffset < height - 1)
            gridList[boidIndex + stride * 4] = make_uint2(cellIndex + width, boidIndex);

        //right-top cell
        if (yOffset != 0 && xOffset != width - 1)
            gridList[boidIndex + stride * 5] = make_uint2(cellIndex - width + 1, boidIndex);

        //right-bot cell
        if (yOffset < height - 1 && xOffset != width - 1)
            gridList[boidIndex + stride * 6] = make_uint2(cellIndex + width + 1, boidIndex);

        //left-top cell
        if (xOffset != 0 && yOffset != 0)
            gridList[boidIndex + stride * 7] = make_uint2(cellIndex - width - 1, boidIndex);

        //left-bot cell
        if (xOffset != 0 && yOffset < height - 1)
            gridList[boidIndex + stride * 8] = make_uint2(cellIndex + width - 1, boidIndex);
    }
}
__global__ void makeLookupTable(uint2* gridList, Boid* boidArray, Cell* cellArray, int* lookUpTable)
{
    int gridListIndex = blockIdx.x * blockDim.x + threadIdx.x;
    if (gridList[gridListIndex].x == HIGHNUMBER)
        return;

    if (gridListIndex < BoidCount)
    {
        if (gridListIndex == 0)
        {
            lookUpTable[gridList[gridListIndex].x] = gridListIndex;
            return;
        }

        if (gridList[gridListIndex].x > gridList[gridListIndex - 1].x)
        {
            lookUpTable[gridList[gridListIndex].x] = gridListIndex;
            return;
        }
    }
    else
    {
        if (gridListIndex == BoidCount)
        {
            lookUpTable[gridList[gridListIndex].x + CELL_ARRAY_SIZE] = gridListIndex;
        }
        if (gridList[gridListIndex].x > gridList[gridListIndex - 1].x)
        {
            lookUpTable[gridList[gridListIndex].x + CELL_ARRAY_SIZE] = gridListIndex;
            return;
        }
    }

}

__device__ float2 calculateSeparation(int* localBoidIDs, int* neighboringBoidIds, int boidIndex, Boid* boids, Boid currentBoid) {
    int closeDx = 0, closeDy = 0;
    //loop through local boids
    for (int i = 0; i < BoidCount; i++)
    {
        if (i != boidIndex && localBoidIDs[i] != -1)
        {
            float distX = currentBoid.position.x - boids[localBoidIDs[i]].position.x;
            float distY = currentBoid.position.y - boids[localBoidIDs[i]].position.y;
            //add only if inside protected range
            if (sqrt(distX * distX + distY * distY) < PROTECTED_RANGE)
            {
                closeDx += distX;
                closeDy += distY;
            }

        }
    }
    //loop through neighboring boids 
    for (int i = 0; i < BoidCount * 8; i++)
    {
        if (localBoidIDs[i] != -1)
        {
            float distX = currentBoid.position.x - boids[neighboringBoidIds[i]].position.x;
            float distY = currentBoid.position.y - boids[neighboringBoidIds[i]].position.y;
            //add only if inside protected range
            if (sqrt(distX * distX + distY * distY) < PROTECTED_RANGE)
            {
                closeDx += distX;
                closeDy += distY;
            }
        }
    }

    return make_float2(closeDx, closeDy);
}
__global__ void calculateBoidLogic(uint2* gridList, Boid* boidArray, Cell* cellArray, int* lookUpTable, float deltaTime)
{
    __shared__ int localBoidIds[MAX_BOIDS_IN_A_CELL];
    __shared__ int neighboringBoidIds[MAX_BOIDS_IN_A_CELL * 8];

    //new aproach to genrating boid arrays (now with more parallel computing :) )
    if (threadIdx.x < MAX_BOIDS_IN_A_CELL) // local array
    {
        //get the index for the gridList
        int lookupIndexLocal = lookUpTable[blockIdx.x];

        //check for -1 meaning there is no boid in the index
        if (lookupIndexLocal == -1)
        {
            localBoidIds[threadIdx.x] = -1;
            return;
        }
        //retrive the cell boid pair from the index ofset by the index of thread
        uint2 cellBoidPairLocal = gridList[lookupIndexLocal + threadIdx.x];

        //check if the retrived pair is still inside the cell
        if (cellBoidPairLocal.x == gridList[lookupIndexLocal].x)
        {
            //if it is then add it to the local boids array
            localBoidIds[threadIdx.x] = cellBoidPairLocal.y;
        }
        else
        {
            //if not assign -1 value as empty
            localBoidIds[threadIdx.x] = -1;
            return;
        }
    }
    else if (threadIdx.x >= MAX_BOIDS_IN_A_CELL) // neighbor array 
    {
        int lookupIndexNeigbor = lookUpTable[blockIdx.x + CELL_ARRAY_SIZE];
        if (lookupIndexNeigbor == -1)
        {
            neighboringBoidIds[threadIdx.x - MAX_BOIDS_IN_A_CELL] = -1;
            return;
        }
        uint2 cellBoidPairNeighbor = gridList[lookupIndexNeigbor + threadIdx.x - MAX_BOIDS_IN_A_CELL];
        if (cellBoidPairNeighbor.x == gridList[lookupIndexNeigbor].x)
        {
            neighboringBoidIds[threadIdx.x - MAX_BOIDS_IN_A_CELL] = cellBoidPairNeighbor.y;
        }
        else
        {
            neighboringBoidIds[threadIdx.x - MAX_BOIDS_IN_A_CELL] = -1;
            return;
        }
    }
    __syncthreads();

    //return unused threads
    if (threadIdx.x >= MAX_BOIDS_IN_A_CELL)
        return;
    if (localBoidIds[threadIdx.x] == -1)
        return;

    int currentBoidId = localBoidIds[threadIdx.x];
    Boid& currentBoid = boidArray[currentBoidId];

    // Fish logic: Update velocity based on separation, alignment, and cohesion rules
    float2 separation = calculateSeparation(localBoidIds, neighboringBoidIds, threadIdx.x, boidArray, currentBoid);

    // Weighting factors for each rule
    float alignmentWeight = 1.0;
    float cohesionWeight = 1.0;

    // Update velocity based on the rules
    //boidArray[threadIdx.x].velocity += separationWeight * separation +
    //    alignmentWeight * alignment +
    //    cohesionWeight * cohesion;

    float edgeAvoidanceFactor = 1;
    float safeDistance = 15.0;      

    //edge detect
    if (currentBoid.position.x < safeDistance)
        separation.x += edgeAvoidanceFactor;
    else if (currentBoid.position.x > WIDTH - safeDistance)
        separation.x -= edgeAvoidanceFactor;

    if (currentBoid.position.y < safeDistance)
        separation.y += edgeAvoidanceFactor;
    else if (currentBoid.position.y > HEIGHT - safeDistance)
        separation.y -= edgeAvoidanceFactor;

    currentBoid.velocity.x += separation.x * AVOIDFACTOR;
    currentBoid.velocity.y += separation.y * AVOIDFACTOR;

    //speed limit
    int speedLimit = 30;
    if (currentBoid.velocity.x > speedLimit)
        currentBoid.velocity.x = speedLimit;
    if (currentBoid.velocity.y > speedLimit)
        currentBoid.velocity.y = speedLimit;

    if (currentBoid.velocity.x < -speedLimit)
        currentBoid.velocity.x = -speedLimit;
    if (currentBoid.velocity.y < -speedLimit)
        currentBoid.velocity.y = -speedLimit;

    //debug code
    //boidArray[localBoidIds[boidIndex]].velocity.x += 1 * AVOIDFACTOR;
    //boidArray[localBoidIds[boidIndex]].velocity.y += 1 * AVOIDFACTOR;

    //printf("Block ID: %d, Thread ID: %d, Boid ID: %d, Position: (%f, %f), Velocity: (%f, %f)\n",
    //    blockIdx.x, threadIdx.x, currentBoidId,
    //    currentBoid.position.x, currentBoid.position.y,
    //    currentBoid.velocity.x, currentBoid.velocity.y);

    currentBoid.position.x += currentBoid.velocity.x * deltaTime;
    currentBoid.position.y += currentBoid.velocity.y * deltaTime;

    __syncthreads();
}
__global__ void setUint2Values(uint2* array, int value, int count) {
    int index = threadIdx.x + blockIdx.x * blockDim.x;
    if (index < count) {
        array[index].x = value;
        array[index].y = value;
    }
}

//shader programs
const char* vertexShaderSource = R"(
    #version 330 core
    layout (location = 0) in vec2 aPos; // Use vec2 for 2D positions
    void main() {
        gl_Position = vec4(aPos.x, aPos.y, 0.0, 1.0);
    }
)";

const char* fragmentShaderSource = R"(
    #version 330 core
    out vec4 FragColor;
    void main() {
        FragColor = vec4(1.0f, 0.5f, 0.2f, 1.0f);
    }
)";


int main()
{
    Boid boidArray[BoidCount];

    // Call the function to initialize Boids
    initializeBoids(WIDTH, HEIGHT, boidArray, BoidCount);

    // Access and use the initialized Boids
    //std::cout << "Boids array____________________________________________________________________________________" << std::endl;
    //for (const auto& boid : boidArray) {
    //    std::cout << "Boid Id: " << boid.Id << ", Position: (" << boid.position.x << ", " << boid.position.y << ")\n";
    //}

    Cell cellArray[CELL_ARRAY_SIZE];

    // Call the function to initialize Cells
    initializeCells(WIDTH, HEIGHT, cellArray, CELL_ARRAY_SIZE);

    // Access and use the initialized Cells
    //std::cout << "Cell array____________________________________________________________________________________" << std::endl;
    //for (const auto& cell : cellArray) {
    //    std::cout << "Cell Id: " << cell.Id << ", Position: (" << cell.position.x << ", " << cell.position.y << ")\n";
    //}

    glfwInit();
    glfwWindowHint(GLFW_CONTEXT_VERSION_MAJOR, 3);
    glfwWindowHint(GLFW_CONTEXT_VERSION_MINOR, 3);
    glfwWindowHint(GLFW_OPENGL_PROFILE, GLFW_OPENGL_CORE_PROFILE);
    cudaSetDevice(0);

    GLFWwindow* window = glfwCreateWindow(WIDTH, HEIGHT, "LearnOpenGL", NULL, NULL);
    if (window == NULL)
    {
        std::cout << "Failed to create GLFW window" << std::endl;
        glfwTerminate();
        return -1;
    }
    glfwMakeContextCurrent(window);

    if (!gladLoadGLLoader((GLADloadproc)glfwGetProcAddress))
    {
        std::cout << "Failed to initialize GLAD" << std::endl;
        return -1;
    }

    glViewport(0, 0, WIDTH, HEIGHT);

    glfwSetFramebufferSizeCallback(window, framebuffer_size_callback);

    unsigned int VAO, VBO;
    glGenVertexArrays(1, &VAO);
    glBindVertexArray(VAO);

    glGenBuffers(1, &VBO);
    glBindBuffer(GL_ARRAY_BUFFER, VBO);
    glBufferData(GL_ARRAY_BUFFER, BoidCount * sizeof(vec2), nullptr, GL_DYNAMIC_DRAW);

    unsigned int vertexShader;
    vertexShader = glCreateShader(GL_VERTEX_SHADER);
    glShaderSource(vertexShader, 1, &vertexShaderSource, NULL);
    glCompileShader(vertexShader);

    unsigned int fragmentShader;
    fragmentShader = glCreateShader(GL_FRAGMENT_SHADER);
    glShaderSource(fragmentShader, 1, &fragmentShaderSource, NULL);
    glCompileShader(fragmentShader);

    int  success;
    char infoLog[512];
    glGetShaderiv(vertexShader, GL_COMPILE_STATUS, &success);
    if (!success)
    {
        glGetShaderInfoLog(vertexShader, 512, NULL, infoLog);
        std::cout << "ERROR::SHADER::VERTEX::COMPILATION_FAILED\n" << infoLog << std::endl;
    }

    glGetShaderiv(fragmentShader, GL_COMPILE_STATUS, &success);
    if (!success)
    {
        glGetShaderInfoLog(fragmentShader, 512, NULL, infoLog);
        std::cout << "ERROR::SHADER::FRAG::COMPILATION_FAILED\n" << infoLog << std::endl;
    }

    unsigned int shaderProgram;
    shaderProgram = glCreateProgram();
    glAttachShader(shaderProgram, vertexShader);
    glAttachShader(shaderProgram, fragmentShader);
    glLinkProgram(shaderProgram);

    glGetProgramiv(shaderProgram, GL_LINK_STATUS, &success);
    if (!success) {
        glGetProgramInfoLog(shaderProgram, 512, NULL, infoLog);
        std::cout << "shader program compliation failed\n" << infoLog << std::endl;
    }
    
    glUseProgram(shaderProgram);
    glDeleteShader(vertexShader);
    glDeleteShader(fragmentShader);

    glVertexAttribPointer(0, 2, GL_FLOAT, GL_FALSE, sizeof(vec2), (void*)0);
    glEnableVertexAttribArray(0);

    Boid* d_boidArray;
    uint2* d_gridList;
    Cell* d_cellArray;
    int* d_lookUpTable;
    std::vector<vec2> renderArray(BoidCount);

    cudaMalloc((void**)&d_boidArray, BoidCount * sizeof(Boid));
    cudaMalloc((void**)&d_gridList, BoidCount * 9 * sizeof(uint2));
    cudaMalloc((void**)&d_cellArray, CELL_ARRAY_SIZE * sizeof(Cell));

    // Copy boidArray and cellArray to device memory
    cudaMemcpy(d_boidArray, boidArray, BoidCount * sizeof(Boid), cudaMemcpyHostToDevice);
    cudaMemcpy(d_cellArray, cellArray, CELL_ARRAY_SIZE * sizeof(Cell), cudaMemcpyHostToDevice);

    //main simulation loop
    double lastFrameTime = glfwGetTime();
    while (!glfwWindowShouldClose(window))
    {
        // Calculate delta time
        double currentFrameTime = glfwGetTime();
        double deltaTime = currentFrameTime - lastFrameTime;
        lastFrameTime = currentFrameTime;

        setUint2Values << <BoidCount, 9 >> > (d_gridList, HIGHNUMBER, BoidCount * 9);
        cudaDeviceSynchronize();

        hashBoids << <1, BoidCount >> > (d_boidArray, d_gridList, d_cellArray, BoidCount, CELL_ARRAY_SIZE, WIDTH, HEIGHT);
        cudaDeviceSynchronize();
        //sort grid list
        SortGridList(d_gridList);
        //std::cout << "sorted array____________________________________________________________________________________" << std::endl;
        //uint2* hostGridList = new uint2[BoidCount * 9];
        //cudaMemcpy(hostGridList, d_gridList, BoidCount * sizeof(uint2) * 9, cudaMemcpyDeviceToHost);
        //int k = 0;
        //for (int i = 0; i < 9; i++)
        //{
        //    for (int j = 0; j < BoidCount; j++)
        //    {
        //        std::cout <<"id: " << k << " cell Id: " << hostGridList[k].x << " boid Id: " << hostGridList[k].y << std::endl;
        //        k++;
        //    }
        //}
        cudaDeviceSynchronize();

        //init lookup table array
        
        checkCudaError(cudaMalloc((void**)&d_lookUpTable, CELL_ARRAY_SIZE * 2 * sizeof(int)), "mallocFailed");
        cudaMemset(d_lookUpTable, -1, CELL_ARRAY_SIZE * 2 * sizeof(int));

        //create look up table
        CreateLookUpTable(d_gridList, d_boidArray, d_cellArray, d_lookUpTable);
        cudaDeviceSynchronize();
        //std::cout << "LookUp table____________________________________________________________________________________" << std::endl;
        //int* h_lookUptable = new int[CELL_ARRAY_SIZE * 2];
        //cudaMemcpy(h_lookUptable, d_lookUpTable, CELL_ARRAY_SIZE * 2 * sizeof(int), cudaMemcpyDeviceToHost);
        //for (int i = 0; i < CELL_ARRAY_SIZE * 2; ++i)
        //    std::cout << "Cell index: " << i % CELL_ARRAY_SIZE << " starts at index: " << h_lookUptable[i] << std::endl;


        CalculateBoidLogic(d_gridList, d_boidArray, d_cellArray, d_lookUpTable, deltaTime);
        cudaDeviceSynchronize();
        cudaMemcpy(boidArray, d_boidArray, BoidCount * sizeof(Boid), cudaMemcpyDeviceToHost);


        // Extract only positions for rendering std::transform!
        for (int i = 0; i < BoidCount; ++i)
        {
            float normalized_x = (2.0f * boidArray[i].position.x / WIDTH) - 1.0f;
            float normalized_y = 1.0f - (2.0f * boidArray[i].position.y / HEIGHT);

            renderArray[i].x = normalized_x;
            renderArray[i].y = normalized_y;
            
            //std::cout << "XPos: " << renderArray[i].x << " YPos: " << renderArray[i].y << std::endl;
         }


        // Update the VBO with the new boid positions
        glBindBuffer(GL_ARRAY_BUFFER, VBO);
        glBufferSubData(GL_ARRAY_BUFFER, 0, BoidCount * sizeof(vec2), renderArray.data());
        //renderArray.clear();

        // OpenGL rendering
        glClearColor(0.2f, 0.3f, 0.3f, 1.0f);
        glClear(GL_COLOR_BUFFER_BIT);

        //shader program and VAO
        glUseProgram(shaderProgram);
        glBindVertexArray(VAO);
        glPointSize(5.0f); // Set point size to make boids larger
        glDrawArrays(GL_POINTS, 0, BoidCount); // Assuming you want to draw points for each boid

        glfwPollEvents();
        glfwSwapBuffers(window);

        //std::cout << std::endl;
        //std::cout << "NEXT FRAME ++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++" << std::endl;
        //std::cout << std::endl;
    }
    // Free allocated memory on the device
    cudaFree(d_boidArray);
    cudaFree(d_gridList);
    cudaFree(d_cellArray);
    cudaFree(d_lookUpTable);
    delete[] boidArray;
    cudaDeviceReset();

    glfwTerminate();
    return 0;
}

void framebuffer_size_callback(GLFWwindow* window, int width, int height)
{
    glViewport(0, 0, width, height);
}
void processInput(GLFWwindow* window)
{
    if (glfwGetKey(window, GLFW_KEY_ESCAPE) == GLFW_PRESS)
        glfwSetWindowShouldClose(window, true);
}
void CalculateBoidLogic(uint2* d_gridList, Boid* d_boidArray, Cell* d_cellArray, int* d_lookUpTable, float deltaTime)
{
    calculateBoidLogic << < CELL_ARRAY_SIZE, MAX_BOIDS_IN_A_CELL * 9>> > (d_gridList, d_boidArray, d_cellArray, d_lookUpTable, deltaTime);
}
void CreateLookUpTable(uint2* d_gridList, Boid* d_boidArray, Cell* d_cellArray, int* d_lookUpTable)
{
    // Determine the total number of threads needed
    //int totalThreads = BoidCount * 9;

    //// Determine the number of threads per block
    //int threadsPerBlock = std::min(totalThreads, 1024);

    //// Determine the number of blocks needed
    //int blocks = (totalThreads + threadsPerBlock - 1) / threadsPerBlock;

    // Launch the kernel with the adjusted configuration
    makeLookupTable << <BoidCount, 9 >> > (d_gridList, d_boidArray, d_cellArray, d_lookUpTable);
}
void SortGridList(uint2* d_gridList)
{
    //thrust::device_ptr<uint2> dev_ptr_gridList(d_gridList);

    //// Sort based on x values
    //thrust::sort(dev_ptr_gridList, dev_ptr_gridList + BoidCount, CompareX());
    //thrust::sort(dev_ptr_gridList + BoidCount, dev_ptr_gridList + BoidCount * 9, CompareX());

    //// Copy the sorted data back to the host if needed
    //cudaMemcpy(d_gridList, thrust::raw_pointer_cast(dev_ptr_gridList), BoidCount * 9 * sizeof(uint2), cudaMemcpyDeviceToDevice);



    thrust::device_vector<uint2> dev_vec_gridList(d_gridList, d_gridList + BoidCount * 9);

    // Sort 
    thrust::sort(dev_vec_gridList.begin(), dev_vec_gridList.begin() + BoidCount, CompareX());
    checkCudaLastError("Sort first half");

    thrust::sort(dev_vec_gridList.begin() + BoidCount, dev_vec_gridList.end(), CompareX());
    checkCudaLastError("Sort second half");

    // Copy the sorted data back to the device
    thrust::copy(dev_vec_gridList.begin(), dev_vec_gridList.end(), d_gridList);
    checkCudaLastError("Copy sorted data back");
    dev_vec_gridList.clear();
}
void initializeBoids(int width, int height, Boid(&boidArray)[BoidCount], int size) {
    // Seed the random number generator
    std::srand(static_cast<unsigned int>(std::time(0)));

    for (int i = 0; i < size; ++i) {
        // Assign random positions within the specified width and height
        float randomX = static_cast<float>(std::rand() % width);
        float randomY = static_cast<float>(std::rand() % height);

        // Assign sequential Ids
        int id = i;

        // Initialize the Boid with the generated values
        boidArray[i] = Boid(id, { randomX, randomY});
    }
}
void initializeCells(int width, int height, Cell(&cellArray)[CELL_ARRAY_SIZE], int cellArraySize) {
    // Calculate the size of each cell based on the grid and number of cells
    float cellWidth = static_cast<float>(width) / static_cast<float>(sqrt(cellArraySize));
    float cellHeight = static_cast<float>(height) / static_cast<float>(sqrt(cellArraySize));
    std::cout << cellWidth << " " << cellHeight << std::endl;
    int k = 0;
    // Loop through the cell array and initialize each cell
    for (int i = 0; i < HEIGHT / cellHeight; i++) {
        for (int j = 0; j < WIDTH / cellWidth; j++)
        {
            float cellX = static_cast<float>(j * cellWidth + cellWidth / 2);
            float cellY = static_cast<float>(i * cellWidth + cellWidth / 2);
            cellArray[k] = Cell{ k, {cellX, cellY} };
            k++;
        }
    }
}

