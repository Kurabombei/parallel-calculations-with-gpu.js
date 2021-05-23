// Importing gpu.js
const { GPU } = require('gpu.js');
const { sqrt, matrix, ones, zeros, transpose, dot, multiply} = require('mathjs');
const {performance} = require('perf_hooks');

// Checks for compability for GPU lib
function checkGpuCompability(){
    console.log("isGPUSupported :" + GPU.isGPUSupported);  // boolean - checks if GPU is in-fact supported
    console.log("isKernelMapSupported :" + GPU.isKernelMapSupported); // boolean - checks if kernel maps are supported
    console.log("isOffscreenCanvasSupported :" + GPU.isOffscreenCanvasSupported); // boolean - checks if offscreen canvas is supported
    console.log("isWebGLSupported :" + GPU.isWebGLSupported); // boolean - checks if WebGL v1 is supported
    console.log("isWebGL2Supported :" + GPU.isWebGL2Supported); // boolean - checks if WebGL v2 is supported
    console.log("isHeadlessGLSupported :" + GPU.isHeadlessGLSupported); //boolean - checks if headlessgl is supported
    console.log("isCanvasSupported :" + GPU.isCanvasSupported); //boolean - checks if canvas is supported
    console.log("isGPUHTMLImageArraySupported :" + GPU.isGPUHTMLImageArraySupported); //boolean - checks if the platform supports HTMLImageArray's
    console.log("isSinglePrecisionSupported :" + GPU.isSinglePrecisionSupported); //boolean - checks if the system supports single precision float 32 values
}




// (function test(){
//     const b = matrix(ones([2, 3]));
//     console.log(sqrt(49));
//
// })()

// ########################  GPU.JS EXAMPLE  ########################
const generateMatrices = (size) => {
    const matrices = [[], [], [], []];
    for (let y = 0; y < size; y++){
        matrices[0].push([]); // matrix A
        matrices[1].push([]); // matrix B
        matrices[2].push([]); // matrix C
        for (let x = 0; x < size; x++){
            matrices[0][y].push(Math.round(Math.random() * size)); // calculation for matrixA
            matrices[1][y].push(Math.round(Math.random() * size)); // calculation for matrixB
            if(x == 0 && y == 0){
                matrices[2][y].push(Math.fround(1 / + (1 ** 2))); // case for infinity
            }else{
                matrices[2][y].push(Math.fround(1 / + (y + x ** 2))); // calculation for matrixC
            }

        }
    }
    return matrices;
}

const generateVectors = (size) => {
    const vectors = [[], []];
    for (let y = 0; y < size; y++){
        vectors[0].push(4 / (y ** 3 + 3)); // vector B
        vectors[1].push(Math.round(Math.random() * size ** 2)); // vector C
    }
    return vectors;
}

function transposeVanilla(array){
    let rows = array.length;
    let cols = array[0].length;
    // Verify array dimensions
    for(let i = 0; i < rows; i++){
        if(array[i].length != cols){
            throw new Error('Array dimension error: must be 2d array withequal row lengths');
        }
    }
    // Create new array
    let transposed = new Array(cols);
    for(let c = 0; c < cols; c++){
        transposed[c] = new Array(rows);
        for(let r = 0; r < rows; r++){
            transposed[c][r] = array[r][c];
        }
    }
    return transposed;
}


const gpu = new GPU();
let matrixSize = 4;

const multiplyMatrix = gpu.createKernel(function(a, b,matrixSize) {
    let sum = 0;
    for (let i = 0; i < matrixSize; i++) {
        sum += a[this.thread.y][i] * b[i][this.thread.x];
    }
    return sum;
}).setOutput([matrixSize, matrixSize])
    .setDynamicArguments(true);

// Level 1
const startTime = performance.now();
const matrices = generateMatrices(matrixSize);// Returning an array with 3 matrices
const vectors = generateVectors(matrixSize);  // Returning an array with 2 vectors
// start timer


console.log("Level 1");
console.log("All matrices:");
console.log(matrices);
console.log("All vectors:");
console.log(vectors);

const TimeLevel1 = performance.now();
const TimeLevel1String = (TimeLevel1 - startTime) + " ms";

// Level 2
const level2_x = gpu.createKernel(function( vectorA,vectorB, matrixSize){
    let res = 0;
    for (let i = 0; i < matrixSize; i++) {
        res = vectorA[this.thread.x] + (4 * vectorB[this.thread.x]);
    }
    return res;
}).setOutput([matrixSize]);

const level2_z = gpu.createKernel(function( matrixB,matrixC, matrixSize){
    let res = 0;
    for (let i = 0; i < matrixSize; i++) {
        res = matrixB[this.thread.y][this.thread.x] + matrixC[this.thread.y][this.thread.x];
    }
    return res;
}).setOutput([matrixSize, matrixSize]);

const x = level2_x(vectors[0], vectors[1], matrixSize);
const z = level2_z(matrices[1], matrices[2], matrixSize);


console.log("Level 2");
console.log("X:");
console.log(x);
console.log("Z:");
console.log(z);

const TimeLevel2 = performance.now();
const TimeLevel2String = (TimeLevel2 - TimeLevel1) + " ms";

// Level 3

const Y1 = multiplyMatrix(matrices[0], vectors[1], matrixSize); // matrixA * vectorB
const Y2 = multiplyMatrix(matrices[0], x, matrixSize); // matrixA * new vector x
const Y3 = multiplyMatrix(matrices[0], z, matrixSize); // matrixA * new matrix z

console.log("Level 3");
console.log("Y1 (matrixA * vectorB):");
console.log(Y1);
console.log("Y2 (matrixA * new vector x):");
console.log(Y2);
console.log("Y3 (matrixA * new matrix z):");
console.log(Y3);

const TimeLevel3 = performance.now();
const TimeLevel3String = (TimeLevel3 - TimeLevel2) + " ms";

// Level 4
// Function for transposing a matrix
function transposeOneLiner(matrix) {
    return matrix[0].map((col, i) => matrix.map(row => row[i]));
}


const level4_h = gpu.createKernel(function(Y3){
    return  Math.pow(Y3[this.thread.y][this.thread.x], 2);
}).setOutput([matrixSize, matrixSize]);

let h = level4_h(Y3);  // (Y3 ** 2) or (matrixA * new matrix z) ** 2;
let d =  multiplyMatrix(h, Y2, matrixSize);  // (Y3 ** 2) * Y2
let g =  multiplyMatrix(Y3, Y1, matrixSize);  // Y3 * Y1
let transposedY2 = transposeVanilla(Y2);
let f =  multiplyMatrix(transposedY2, Y1, matrixSize);  // Y2.T * Y1

console.log("Level 4");
console.log("h (Y3 ** 2) or (matrixA * new matrix z) ** 2:");
console.log(h);
console.log("d (Y3 ** 2) * Y2:");
console.log(d);
console.log("g (Y3 * Y1):");
console.log(g);
console.log("f (Y2.T * Y1):");
console.log(f);


const TimeLevel4 = performance.now();
const TimeLevel4String = (TimeLevel4 - TimeLevel3) + " ms";

// Level 5

let j = multiplyMatrix(g, transposeVanilla(d), matrixSize);
let k = multiplyMatrix(f, g, matrixSize);

console.log("Level 5");
console.log("j:");
console.log(j);

console.log("k:");
console.log(k);

const TimeLevel5 = performance.now();
const TimeLevel5String = (TimeLevel5 - TimeLevel4) + " ms";

// Level 6

let l = transposeVanilla(level2_z(k, g, matrixSize));
let result = multiplyMatrix(l, Y3, matrixSize);

console.log("Level 6");
console.log("l:");
console.log(l);

console.log("result:");
console.log(result);




const endTime = performance.now();
const TimeLevel6String = (endTime - TimeLevel5) + " ms";
this.gpuTime = (endTime - startTime) + " ms";
console.log("Level 1 time: " + TimeLevel1String);
console.log("Level 2 time: " + TimeLevel2String);
console.log("Level 3 time: " + TimeLevel3String);
console.log("Level 4 time: " + TimeLevel4String);
console.log("Level 5 time: " + TimeLevel5String);
console.log("Level 6 time: " + TimeLevel6String);
console.log("Total GPU time : "+ this.gpuTime);

// console.log(out); // Logs the element at the xth row and the yth column of the matrix
// console.log(out[10][12]); // Logs the element at the 10th row and the 12th column of the output matrix

// ########################  GPU.JS EXAMPLE  ########################
// Calling a main function
function main(){

    // A function to multiply matrices on cpu
    function cpuMultiplyMatrix() {
        const startTime = performance.now();
        console.time('cpu-multiply-matrix');
        const a = this.matrices[0];
        const b = this.matrices[1];

        let productRow = Array.apply(null, new Array(this.matrixSize)).map(Number.prototype.valueOf, 0);
        let product = new Array(this.matrixSize);
        for (let p = 0; p < this.matrixSize; p++) {
            product[p] = productRow.slice();
        }

        for (let i = 0; i < this.matrixSize; i++) {
            for (let j = 0; j < this.matrixSize; j++) {
                for (let k = 0; k < this.matrixSize; k++) {
                    product[i][j] += a[i][k] * b[k][j];
                }
            }
        }

        console.timeEnd('cpu-multiply-matrix');
        const endTime = performance.now();
        this.cpuTime = (endTime - startTime) + " ms";
        this.cpuProduct = product;
    }
    // A function to multiply matrices on gpu
    function gpuMultiplyMatrix() {
        const gpu = new GPU();
        const multiplyMatrix = gpu.createKernel(function (a, b, matrixSize) {
            let sum = 0;

            for (let i = 0; i < matrixSize; i++) {
                sum += a[this.thread.y][i] * b[i][this.thread.x];
            }
            return sum;
        }).setOutput([this.matrixSize, this.matrixSize]);
        const startTime = performance.now();
        const resultMatrix = multiplyMatrix(this.matrices[0],  this.matrices[1], this.matrixSize);

        const endTime = performance.now();
        this.gpuTime = (endTime - startTime) + " ms";

        console.log("GPU TIME : "+ this.gpuTime);
        this.gpuProduct = resultMatrix;
    }

    // Creating kernels for parallel calculations

    // As we are only creating matrices here, no need to create them on gpu,
    // because our gpu should handle only the needed calculations (matrix multiplication and so on)

}
