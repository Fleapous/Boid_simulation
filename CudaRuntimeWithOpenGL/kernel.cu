// Include CUDA headers first
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

#define BOID_COUNT 15000
#define CELL_ARRAY_SIZE 5625

#define WIDTH 1500
#define HEIGHT 1500

#define HIGHNUMBER 1000000
#define MAX_BOIDS_IN_A_CELL 32 

//boid logic parameters
// general 
#define MAX_SPEED 60
#define MIN_SPEED 20
#define EDGE_RANGE 100
#define EDGE_AVOIDANCE_FACTOR 3.2


//seperation 
#define PROTECTED_RANGE 4
#define AVOIDFACTOR 2.5

//Alignment & Cohesion
#define VISIBLE_RANGE 35
#define MATCHINGFACTOR 0.01
#define CENTERINGFACTOR 0.08

uint2 gridList[BOID_COUNT];

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
struct CompareX {
    __host__ __device__
        bool operator()(const uint2& a, const uint2& b) const {
        return a.x < b.x;
    }
};

void initializeBoids(float(&PosX)[BOID_COUNT], float(&PosY)[BOID_COUNT], float(&Vx)[BOID_COUNT], float(&Vy)[BOID_COUNT]);
void SortGridList(uint2* d_gridList);
void initializeCells(int width, int height, Cell(&cellArray)[CELL_ARRAY_SIZE], int cellArraySize);
void CreateLookUpTable(uint2* d_gridList, int* d_lookUpTable);
void CalculateBoidLogic(uint2* d_gridList, float* PosX, float* PosY, float* Vx, float* Vy, Cell* d_cellArray, int* d_lookUpTable, float deltaTime);
void framebuffer_size_callback(GLFWwindow* window, int width, int height);
void processInput(GLFWwindow* window);
void SetUnit2Values(uint2* array);
void HashBoids(float* PosX, float* PosY, uint2* d_gridList);
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
        
        exit(EXIT_FAILURE);
    }
}

__global__ void hashBoids(float* PosX, float* PosY, uint2* gridList) {
    int boidIndex = threadIdx.x + blockIdx.x * blockDim.x;

    if (boidIndex < BOID_COUNT) {
        int boidXPos = static_cast<int>(static_cast<int>(PosY[boidIndex] / (HEIGHT / sqrt((float)CELL_ARRAY_SIZE))) * sqrt((float)CELL_ARRAY_SIZE)); // only need the integer part 
        int boidYPos = static_cast<int>(static_cast<int>(PosX[boidIndex] / (WIDTH / sqrt((float)CELL_ARRAY_SIZE))));
        int cellIndex = boidXPos + boidYPos;
        gridList[boidIndex] = make_uint2(cellIndex, boidIndex);

        //printf("boidPosX: %f, boidPosY: %f, boidXPos: %d, boidYPos: %d, cellIndex: %d\n",
        //    boidArray[boidIndex].position.x,boidArray[boidIndex].position.y, boidXPos, boidYPos, cellIndex);
        //checking if cell is a corner or a side
        // temporarry 
        int height = static_cast<int>(sqrt((float)CELL_ARRAY_SIZE));
        int width = height;

        int xOffset = cellIndex % width;
        int yOffset = static_cast<int>((cellIndex - xOffset) / height);

        //adding neighbors 
        int stride = BOID_COUNT;
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
__global__ void makeLookupTable(uint2* gridList, int* lookUpTable)
{
    int gridListIndex = blockIdx.x * blockDim.x + threadIdx.x;
    if (gridList[gridListIndex].x == HIGHNUMBER)
        return;

    if (gridListIndex < BOID_COUNT)
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
        if (gridListIndex == BOID_COUNT)
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

__device__ float2 calculateSeparation(int* localBoidIDs, int* neighboringBoidIds, int boidIndex, float* PosX, float* PosY, float currentPosX, float currentPosY) {
    float closeDx = 0, closeDy = 0;
    //loop through local boids
    for (int i = 0; i < MAX_BOIDS_IN_A_CELL; i++)
    {
        if (i != boidIndex && localBoidIDs[i] != -1)
        {
            //printf("current boid vs compared boid: (%d, %d)\n", currentBoid.Id, boids[localBoidIDs[i]].Id);
            float distX = currentPosX - PosX[localBoidIDs[i]];
            float distY = currentPosY - PosY[localBoidIDs[i]];
            //add only if inside protected range
            if (sqrt(distX * distX + distY * distY) < PROTECTED_RANGE)
            {
                closeDx += distX;
                closeDy += distY;
            }

        }
    }
    //loop through neighboring boids 
    for (int i = 0; i < MAX_BOIDS_IN_A_CELL * 8; i++)
    {
        if (localBoidIDs[i] != -1 && i < MAX_BOIDS_IN_A_CELL * 8)
        {
            float distX = currentPosX - PosX[neighboringBoidIds[i]];
            float distY = currentPosY - PosY[neighboringBoidIds[i]];
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
__device__ float2 calculateAlignment(int* localBoidIDs, int* neighboringBoidIds, int boidIndex, float* PosX, float* PosY, float* Vx, float* Vy, float currentPosX, float currentPosY) // Look into range calculation!!!!!!!!!!!!!!
{
    float xvelAvg = 0, yvelAvg = 0;
    int neighboring_boids = 0;
    //loop through local boids
    for (int i = 0; i < BOID_COUNT; i++)
    {
        if (i >= MAX_BOIDS_IN_A_CELL)
            break;

        if (i != boidIndex && localBoidIDs[i] != -1)
        {
            //printf("current boid vs compared boid: (%d, %d)\n", currentBoid.Id, boids[localBoidIDs[i]].Id);
            float distX = currentPosX - PosX[localBoidIDs[i]];
            float distY = currentPosY - PosY[localBoidIDs[i]];
            //add only if inside protected range
            if (sqrt(distX * distX + distY * distY) < VISIBLE_RANGE)
            {
                xvelAvg += Vx[localBoidIDs[i]];
                yvelAvg += Vy[localBoidIDs[i]];
                neighboring_boids++;
            }

        }
    }
    //loop through neighboring boids 
    for (int i = 0; i < BOID_COUNT * 8; i++)
    {
        if (i >= MAX_BOIDS_IN_A_CELL * 8)
            break;
        if (localBoidIDs[i] != -1 && i < MAX_BOIDS_IN_A_CELL * 8)
        {
            float distX = currentPosX - PosX[neighboringBoidIds[i]];
            float distY = currentPosY - PosY[neighboringBoidIds[i]];
            //add only if inside protected range
            if (sqrt(distX * distX + distY * distY) < VISIBLE_RANGE)
            {
                xvelAvg += Vx[neighboringBoidIds[i]];
                yvelAvg += Vy[neighboringBoidIds[i]];
                neighboring_boids++;
            }
        }
    }

    if (neighboring_boids > 0)
    {
        xvelAvg = xvelAvg / neighboring_boids;
        yvelAvg = yvelAvg / neighboring_boids;
    }


    return make_float2(xvelAvg, yvelAvg);
}
__device__ float2 calculateCohesion(int* localBoidIDs, int* neighboringBoidIds, int boidIndex, float* PosX, float* PosY, float currentPosX, float currentPosY)
{
    float xposAvg = 0, yposAvg = 0;
    int neighboring_boids = 0;
    //loop through local boids
    for (int i = 0; i < BOID_COUNT; i++)
    {
        if (i >= MAX_BOIDS_IN_A_CELL)
            break;

        if (i != boidIndex && localBoidIDs[i] != -1)
        {
            //printf("current boid vs compared boid: (%d, %d)\n", currentBoid.Id, boids[localBoidIDs[i]].Id);
            float distX = currentPosX - PosX[localBoidIDs[i]];
            float distY = currentPosY - PosY[localBoidIDs[i]];
            //add only if inside protected range
            if (sqrt(distX * distX + distY * distY) < VISIBLE_RANGE)
            {
                xposAvg += PosX[localBoidIDs[i]];
                yposAvg += PosY[localBoidIDs[i]];
                neighboring_boids++;
            }

        }
    }
    //loop through neighboring boids 
    for (int i = 0; i < BOID_COUNT * 8; i++)
    {
        if (i >= MAX_BOIDS_IN_A_CELL * 8)
            break;
        if (localBoidIDs[i] != -1 && i < MAX_BOIDS_IN_A_CELL * 8)
        {
            float distX = currentPosX - PosX[neighboringBoidIds[i]];
            float distY = currentPosY - PosY[neighboringBoidIds[i]];
            //add only if inside protected range
            if (sqrt(distX * distX + distY * distY) < VISIBLE_RANGE)
            {
                xposAvg += PosX[neighboringBoidIds[i]];
                yposAvg += PosY[neighboringBoidIds[i]];
                neighboring_boids++;
            }
        }
    }

    if (neighboring_boids > 0)
    {
        xposAvg = xposAvg / neighboring_boids;
        yposAvg = yposAvg / neighboring_boids;
    }


    return make_float2(xposAvg, yposAvg);
}
__global__ void calculateBoidLogic(uint2* gridList, float* PosX, float* PosY, float* Vx, float* Vy, Cell* cellArray, int* lookUpTable, float deltaTime)
{
    __shared__ int localBoidIds[MAX_BOIDS_IN_A_CELL];
    __shared__ int neighboringBoidIds[MAX_BOIDS_IN_A_CELL * 8];

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
        //checking for bounds of local Boids in gridList array
        if (lookupIndexLocal + threadIdx.x >= BOID_COUNT) 
        {
            localBoidIds[threadIdx.x] = -1;
            return;
        }
        //retrive the cell boid pair from the index ofset by the index of thread
        //printf("Block: %d, Thread: %d, index: %d max 100\n",
        //    blockIdx.x, threadIdx.x, lookupIndexLocal + threadIdx.x);
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

        //printf("Block: %d, Thread: %d, GridList.x: %u, GridList.y: %u\n",
        //    blockIdx.x, threadIdx.x, gridList[lookupIndexLocal + threadIdx.x].x, gridList[lookupIndexLocal + threadIdx.x].y);

    }

    if (threadIdx.x >= MAX_BOIDS_IN_A_CELL) // neighbor array max thread 863 
    {
        int lookupIndexNeigbor = lookUpTable[blockIdx.x + CELL_ARRAY_SIZE]; // 732 is max number 
        if (lookupIndexNeigbor == -1)
        {
            neighboringBoidIds[threadIdx.x - MAX_BOIDS_IN_A_CELL] = -1;
            return;
        }

        //checking for bouds of the gridList
        if (lookupIndexNeigbor + threadIdx.x - MAX_BOIDS_IN_A_CELL >= BOID_COUNT * 9) // if it is more than 1035 
        {
            neighboringBoidIds[threadIdx.x - MAX_BOIDS_IN_A_CELL] = -1;
            return;
        }
        //printf("Block: %d, Thread: %d, index: %d Max 899\n",
        //    blockIdx.x, threadIdx.x, lookupIndexNeigbor + threadIdx.x - MAX_BOIDS_IN_A_CELL);
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

        //printf("Block: %d, Thread: %d, GridList.x: %u, GridList.y: %u\n",
        //    blockIdx.x, threadIdx.x, gridList[lookupIndexNeigbor + threadIdx.x - MAX_BOIDS_IN_A_CELL].x, gridList[lookupIndexNeigbor + threadIdx.x - MAX_BOIDS_IN_A_CELL].y);
    }
    __syncthreads();

    //return unused threads
    if (threadIdx.x >= MAX_BOIDS_IN_A_CELL)
        return;
    if (localBoidIds[threadIdx.x] == -1)
        return;

    int currentBoidId = localBoidIds[threadIdx.x];
    //Boid& currentBoid = boidArray[currentBoidId];

    float& currentPosX = PosX[currentBoidId];
    float& currentPosY = PosY[currentBoidId];
    float& currentVx = Vx[currentBoidId];
    float& currentVy = Vy[currentBoidId];

    //printf("block and thread %d, %d. currentBoidID: %d. BoidsID %d\n", blockIdx.x, threadIdx.x, currentBoidId, currentBoid.Id);
    // Fish logic: Update velocity based on separation, alignment, and cohesion rules
    float2 separation = calculateSeparation(localBoidIds, neighboringBoidIds, threadIdx.x, PosX, PosY, currentPosX, currentPosY);
    float2 alignment = calculateAlignment(localBoidIds, neighboringBoidIds, threadIdx.x, PosX, PosY, Vx, Vy, currentPosX, currentPosY);
    float2 cohesion = calculateCohesion(localBoidIds, neighboringBoidIds, threadIdx.x, PosX, PosY, currentPosX, currentPosY);

    // Apply separation
    currentVx += separation.x * AVOIDFACTOR;
    currentVy += separation.y * AVOIDFACTOR;

    // Apply alignment
    currentVx += (alignment.x - currentVx) * MATCHINGFACTOR;
    currentVy += (alignment.y - currentVy) * MATCHINGFACTOR;

    // Apply Cohesion
    currentVx += (cohesion.x - currentPosX) * CENTERINGFACTOR;
    currentVy += (cohesion.y - currentPosY) * CENTERINGFACTOR;

    // Edge detection
    float2 edgeAvoidance = make_float2(0.0f, 0.0f);

    if (currentPosX < EDGE_RANGE)
        edgeAvoidance.x += EDGE_AVOIDANCE_FACTOR;
    if (currentPosX > WIDTH - EDGE_RANGE)
        edgeAvoidance.x -= EDGE_AVOIDANCE_FACTOR;

    if (currentPosY < EDGE_RANGE)
        edgeAvoidance.y += EDGE_AVOIDANCE_FACTOR;
    if (currentPosY > HEIGHT - EDGE_RANGE)
        edgeAvoidance.y -= EDGE_AVOIDANCE_FACTOR;

    // Apply edge avoidance
    currentVx += edgeAvoidance.x;
    currentVy += edgeAvoidance.y;

    float speed = static_cast<float>(sqrt(currentVx * currentVx + currentVy * currentVy));

    //Enforce min and max speeds
    if (speed < MIN_SPEED)
    {
        currentVx = (currentVx / speed) * MIN_SPEED;
        currentVy = (currentVy / speed) * MIN_SPEED;
    }
    if (speed > MAX_SPEED)
    {
        currentVx = (currentVx / speed) * MAX_SPEED;
        currentVy = (currentVy / speed) * MAX_SPEED;
    }

    currentPosX += currentVx * deltaTime;
    currentPosY += currentVy * deltaTime;

    __syncthreads();
}
__global__ void setUint2Values(uint2* array) {
    int index = threadIdx.x + blockIdx.x * blockDim.x;
    if (index < BOID_COUNT * 9) {
        array[index].x = HIGHNUMBER;
        array[index].y = HIGHNUMBER;
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
        FragColor = vec4(0.1411764705882353f, 0.41568627450980394f, 0.45098039215686275f, 1.0f);
    }
)";


// Global variables
double lastTime = 0.0;
int frameCount = 0;

// Function to calculate and display FPS
void calculateFPS(GLFWwindow* window) {
    double currentTime = glfwGetTime();
    double deltaTime = currentTime - lastTime;
    frameCount++;

    if (deltaTime >= 1.0) {
        double fps = static_cast<double>(frameCount) / deltaTime;
        std::cout << "FPS: " << fps << std::endl;

        frameCount = 0;
        lastTime = currentTime;
    }
}

int main()
{
    float PosX[BOID_COUNT];
    float PosY[BOID_COUNT];

    float Vx[BOID_COUNT];
    float Vy[BOID_COUNT];

    // Call the function to initialize Boids
    initializeBoids(PosY, PosX, Vx, Vy);

    //Access and use the initialized Boids
   //std::cout << "Boids array____________________________________________________________________________________" << std::endl;
   //for (int i = 0; i < BOID_COUNT; i++) {
   //    std::cout << "Boid Id: " << i << ", Position: (" << PosX[i] << ", " << PosY[i] << ")\n";
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
    glBufferData(GL_ARRAY_BUFFER, BOID_COUNT * sizeof(vec2), nullptr, GL_DYNAMIC_DRAW);

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

    float* d_PosX;
    float* d_PosY;
    float* d_Vx;
    float* d_Vy;

    uint2* d_gridList;
    Cell* d_cellArray;
    int* d_lookUpTable;
    std::vector<vec2> renderArray(BOID_COUNT);

    cudaMalloc((void**)&d_PosX, BOID_COUNT * sizeof(float));
    cudaMalloc((void**)&d_PosY, BOID_COUNT * sizeof(float));
    cudaMalloc((void**)&d_Vx, BOID_COUNT * sizeof(float));
    cudaMalloc((void**)&d_Vy, BOID_COUNT * sizeof(float));

    cudaMalloc((void**)&d_gridList, BOID_COUNT * 9 * sizeof(uint2));
    cudaMalloc((void**)&d_cellArray, CELL_ARRAY_SIZE * sizeof(Cell));

    // Copy boidArray and cellArray to device memory
    checkCudaError(cudaMalloc((void**)&d_lookUpTable, CELL_ARRAY_SIZE * 2 * sizeof(int)), "mallocFailed");
   
    cudaMemcpy(d_PosX, PosX, BOID_COUNT * sizeof(float), cudaMemcpyHostToDevice);
    cudaMemcpy(d_PosY, PosY, BOID_COUNT * sizeof(float), cudaMemcpyHostToDevice);
    cudaMemcpy(d_Vx, Vx, BOID_COUNT * sizeof(float), cudaMemcpyHostToDevice);
    cudaMemcpy(d_Vy, Vy, BOID_COUNT * sizeof(float), cudaMemcpyHostToDevice);

    cudaMemcpy(d_cellArray, cellArray, CELL_ARRAY_SIZE * sizeof(Cell), cudaMemcpyHostToDevice);

    //main simulation loop
    double lastFrameTime = glfwGetTime();
    while (!glfwWindowShouldClose(window))
    {
        // Calculate delta time
        double currentFrameTime = glfwGetTime();
        double deltaTime = currentFrameTime - lastFrameTime;
        lastFrameTime = currentFrameTime;

        SetUnit2Values(d_gridList);
        HashBoids(d_PosX, d_PosY, d_gridList);
        SortGridList(d_gridList);

        //std::cout << "sorted array____________________________________________________________________________________" << std::endl;
        //uint2* hostGridList = new uint2[BOID_COUNT * 9];
        //cudaMemcpy(hostGridList, d_gridList, BOID_COUNT * sizeof(uint2) * 9, cudaMemcpyDeviceToHost);
        //int k = 0;
        //for (int i = 0; i < 9; i++)
        //{
        //    for (int j = 0; j < BOID_COUNT; j++)
        //    {
        //        std::cout <<"id: " << k << " cell Id: " << hostGridList[k].x << " boid Id: " << hostGridList[k].y << std::endl;
        //        k++;
        //    }
        //}

        cudaDeviceSynchronize();
        cudaMemset(d_lookUpTable, -1, CELL_ARRAY_SIZE * 2 * sizeof(int));

        //create look up table
        CreateLookUpTable(d_gridList, d_lookUpTable);
        cudaDeviceSynchronize();

        //std::cout << "LookUp table____________________________________________________________________________________" << std::endl;
        //int* h_lookUptable = new int[CELL_ARRAY_SIZE * 2];
        //cudaMemcpy(h_lookUptable, d_lookUpTable, CELL_ARRAY_SIZE * 2 * sizeof(int), cudaMemcpyDeviceToHost);
        //for (int i = 0; i < CELL_ARRAY_SIZE * 2; ++i)
        //    std::cout << "index: " << i << " Cell index: " << i % CELL_ARRAY_SIZE << " starts at index: " << h_lookUptable[i] << std::endl;

        CalculateBoidLogic(d_gridList, d_PosX, d_PosY, d_Vx, d_Vy, d_cellArray, d_lookUpTable, deltaTime);
        cudaDeviceSynchronize();

        //std::cout << "reseted array____________________________________________________________________________________" << std::endl;
        //uint2* tmp = new uint2[BOID_COUNT * 9];
        //cudaMemcpy(tmp, d_gridList, BOID_COUNT * sizeof(uint2) * 9, cudaMemcpyDeviceToHost);
        //int k2 = 0;
        //for (int i = 0; i < 9; i++)
        //{
        //    for (int j = 0; j < BOID_COUNT; j++)
        //    {
        //        std::cout << "id: " << k2 << " cell Id: " << tmp[k2].x << " boid Id: " << tmp[k2].y << std::endl;
        //        k2++;
        //    }
        //}
        //delete tmp;

        cudaMemcpy(PosX, d_PosX, BOID_COUNT * sizeof(float), cudaMemcpyDeviceToHost);
        cudaMemcpy(PosY, d_PosY, BOID_COUNT * sizeof(float), cudaMemcpyDeviceToHost);

         //Extract only positions for rendering std::transform!
        for (int i = 0; i < BOID_COUNT; ++i)
        {
            //std::cout << "positions: " << PosX[i] << ", " << PosY[i] << std::endl;
           
            float normalized_x = (2.0f * PosX[i] / WIDTH) - 1.0f;
            float normalized_y = 1.0f - (2.0f * PosY[i] / HEIGHT);

            renderArray[i].x = normalized_x;
            renderArray[i].y = normalized_y;

            //std::cout << "XPos: " << renderArray[i].x << " YPos: " << renderArray[i].y << std::endl;
        }

        // Update the VBO with the new boid positions
        glBindBuffer(GL_ARRAY_BUFFER, VBO);
        glBufferSubData(GL_ARRAY_BUFFER, 0, BOID_COUNT * sizeof(vec2), renderArray.data());
        //renderArray.clear();

        // OpenGL rendering
        
        glClearColor(0.08627450980392157f, 0.058823529411764705f, 0.1607843137254902f, 1.0f);
        glClear(GL_COLOR_BUFFER_BIT);

        //shader program and VAO
        glUseProgram(shaderProgram);
        glBindVertexArray(VAO);
        glPointSize(2.0f);
        glDrawArrays(GL_POINTS, 0, BOID_COUNT); 

        calculateFPS(window);
        glfwPollEvents();
        glfwSwapBuffers(window);

        //std::cout << std::endl;
        //std::cout << "NEXT FRAME ++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++" << std::endl;
        //std::cout << std::endl;
    }
    // Free allocated memory on the device
    cudaFree(d_gridList);
    cudaFree(d_cellArray);
    cudaFree(d_lookUpTable);
    cudaFree(d_PosX);
    cudaFree(d_PosY);
    cudaFree(d_Vx);
    cudaFree(d_Vy);
    delete[] PosX;
    delete[] PosY;
    delete[] Vx;
    delete[] Vy;
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
void CalculateBoidLogic(uint2* d_gridList, float* PosX, float* PosY, float* Vx, float* Vy, Cell* d_cellArray, int* d_lookUpTable, float deltaTime)
{
    calculateBoidLogic << < CELL_ARRAY_SIZE, MAX_BOIDS_IN_A_CELL * 9 >> > (d_gridList, PosX, PosY, Vx, Vy, d_cellArray, d_lookUpTable, deltaTime);

    // Check for errors
    checkCudaLastError("kernel launch");

    // Wait for the kernel to finish
    cudaDeviceSynchronize();
}
void CreateLookUpTable(uint2* d_gridList, int* d_lookUpTable)
{
    int threadsPerBlock = 256;

    // Calculate the number of blocks needed based on the total number of elements
    int numBlocks = (BOID_COUNT * 9 + threadsPerBlock - 1) / threadsPerBlock;

    // Launch the kernel with the adjusted configuration
    makeLookupTable << <BOID_COUNT, 9>> > (d_gridList, d_lookUpTable);
}
void SortGridList(uint2* d_gridList)
{
    thrust::device_vector<uint2> dev_vec_gridList(d_gridList, d_gridList + BOID_COUNT * 9);

    // Sort 
    thrust::sort(dev_vec_gridList.begin(), dev_vec_gridList.begin() + BOID_COUNT, CompareX());
    checkCudaLastError("Sort first half");

    thrust::sort(dev_vec_gridList.begin() + BOID_COUNT, dev_vec_gridList.end(), CompareX());
    checkCudaLastError("Sort second half");

    // Copy the sorted data back to the device
    thrust::copy(dev_vec_gridList.begin(), dev_vec_gridList.end(), d_gridList);
    checkCudaLastError("Copy sorted data back");
    dev_vec_gridList.clear();
}
void SetUnit2Values(uint2* d_gridList)
{
    // Choose a block size that is a multiple of 32 (warp size)
    int threadsPerBlock = 256; 

    // Calculate the number of blocks needed based on the total number of elements
    int numBlocks = (BOID_COUNT * 9 + threadsPerBlock - 1) / threadsPerBlock;

    // Launch the kernel with the specified configuration
    setUint2Values << <numBlocks, threadsPerBlock >> > (d_gridList);
    cudaDeviceSynchronize();
}
void HashBoids(float* PosX, float* PosY, uint2* d_gridList)
{
    // Choose a block size that is a multiple of 32 (warp size) 
    int threadsPerBlock = 256; 

    // Calculate the number of blocks needed based on the total number of elements
    int numBlocks = (BOID_COUNT * 9 + threadsPerBlock - 1) / threadsPerBlock;

    hashBoids << <numBlocks, threadsPerBlock >> > (PosX, PosY, d_gridList);
    cudaDeviceSynchronize();
}
void initializeBoids(float(&PosX)[BOID_COUNT], float(&PosY)[BOID_COUNT], float(&Vx)[BOID_COUNT], float(&Vy)[BOID_COUNT]) {
    // Seed the random number generator
    std::srand(static_cast<unsigned int>(std::time(0)));

    for (int i = 0; i < BOID_COUNT; ++i) {
        // Assign random positions 
        PosX[i] = static_cast<float>(std::rand() % WIDTH);
        PosY[i] = static_cast<float>(std::rand() % HEIGHT);

        Vx[i] = static_cast<float>(std::rand()) / RAND_MAX * (MAX_SPEED - MIN_SPEED) + MIN_SPEED;
        Vy[i] = static_cast<float>(std::rand()) / RAND_MAX * (MAX_SPEED - MIN_SPEED) + MIN_SPEED;
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

