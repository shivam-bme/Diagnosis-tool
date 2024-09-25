const express = require('express');
const multer = require('multer');
const sharp = require('sharp');
const tf_c=require('@tensorflow/tfjs-converter')
const tf = require('@tensorflow/tfjs-node');
const ort = require('onnxruntime-node');
const { createCanvas, loadImage } = require('canvas');
const fs = require('fs');
const path = require('path');
const app = express();
const zlib = require('zlib');

const dicomParser = require('dicom-parser');
const PNG  = require('pngjs').PNG;
//const { DataType } = require('@tensorflow/tfjs-converter/dist/data/compiled_api');
const port = 3000;

// Set up multer for handling file uploads
const storage = multer.memoryStorage();

const upload = multer({
    storage: multer.memoryStorage(),
    fileFilter: (req, file, cb) => {
        const allowedMimeTypes = [
            'image/png',
            'image/jpeg',
            'image/tiff',
            'image/tif',
            'application/dicom',
            'application/octet-stream',
            'application/x-mha'
        ];
        console.log(file.mimetype);
        if (allowedMimeTypes.includes(file.mimetype)) {
            cb(null, true);
            console.log('Image allowed')
        } else {
            cb(new Error('Invalid file type. Only DICOM, metal image(MHA), TIFF, PNG, and JPEG files are allowed....'));
        }
    }
});

console.log(__dirname, "next out :",(path.join(__dirname,'/public')) );
// Serve static files
app.use(express.static(path.join(__dirname,'/public')));
app.set('view egnine','ejs');

// Route to serve index.html
app.get('/', (req, res) => {
    res.sendFile(path.join(__dirname, '/index.html'));
});
// conversion code 
const convertTiffToPng = async (inputBuffer) => {
    const outputBuffer = await sharp(inputBuffer).png().toBuffer();
    return outputBuffer;
};

function parseMHA(imageBuffer) {
    const textDecoder = new TextDecoder('utf-8');
    const text = textDecoder.decode(imageBuffer);
    

    // Separate the header and binary data
    const headerEndIndex = text.indexOf('\n\n');
    if (headerEndIndex === -1) {
        throw new Error('Invalid MHA file format: Header not properly terminated.');
    }

    const headerText = text.substring(0, headerEndIndex).trim();
    const headerLines = headerText.split('\n');
    const metadata = {};
    //console.log(headerLines);
    headerLines.forEach(line => {
        //const [key, value] = line.split('=');
        //console.log(value, 'key',key)
        //metadata[key.trim()] = value.trim();
    const [key, value] = line.split('=');
    if (key && value) {
            metadata[key.trim()] = value.trim();    
    } else {
        console.warn(`Skipping malformed header line: ${line}`);
    }
    });
    console.log(headerEndIndex)
    // Binary data starts after the header
    const binaryDataStart = headerEndIndex + 2;
    const binaryDataBuffer =  new Uint8Array(imageBuffer.slice(binaryDataStart));

    // Decompress if needed
    let pixelData;
    if (metadata.CompressedData === 'True') {
        pixelData = zlib.inflateSync(binaryDataBuffer);
    } else {
        pixelData = binaryDataBuffer;
    }

    const width = parseInt(metadata.DimSize.split(' ')[0], 10);
    const height = parseInt(metadata.DimSize.split(' ')[1], 10);

    // Assuming MET_USHORT means 16-bit unsigned short integers
    const data = new Uint16Array(pixelData.buffer, pixelData.byteOffset, pixelData.length / 2);

    return { width, height, data };
}


function convertMhaToPng(inputBuffer) {
    const { width, height, data } = parseMHA(inputBuffer);

    const png = new PNG({ width, height });

    // Normalize the data
    const min = Math.min(...data);
    const max = Math.max(...data);

    for (let i = 0; i < data.length; i++) {
        const normalizedValue = ((data[i] - min) / (max - min)) * 255;
        png.data[i * 4] = normalizedValue;     // R
        png.data[i * 4 + 1] = normalizedValue; // G
        png.data[i * 4 + 2] = normalizedValue; // B
        png.data[i * 4 + 3] = 255;             // A
    }

    png.pack().pipe();
}



const convertDicomToPng = async (inputBuffer) => {
    const outputDir = path.join(__dirname, 'converted_images');
    if (!fs.existsSync(outputDir)) {
        fs.mkdirSync(outputDir);
    }

    const dataSet = dicomParser.parseDicom(inputBuffer);
    const pixelData = new Uint8Array(dataSet.byteArray.buffer, dataSet.elements.x7fe00010.dataOffset, dataSet.elements.x7fe00010.length);
    const width = dataSet.uint16('x00280011');
    const height = dataSet.uint16('x00280010');

    const png = new PNG({ width, height, colorType: 0 });
    for (let y = 0; y < height; y++) {
        for (let x = 0; x < width; x++) {
            const value = pixelData[y * width + x];
            const idx = (y * width + x) * 4;
            png.data[idx] = value;
            png.data[idx + 1] = value;
            png.data[idx + 2] = value;
            png.data[idx + 3] = 255;
        }
    }

    const pngPath = path.join(outputDir, 'converted.png');
    await new Promise((resolve, reject) => {
        png.pack().pipe(fs.createWriteStream(pngPath)).on('finish', resolve).on('error', reject);
    });

    return pngPath;
};

// Function to handle model prediction
const handleModelPrediction = async (req, res, modelPath, htmlPath) => {
    try {
        console.log('it hit');
        // Load the image into a tensor
        let imgBuffer;

        // Convert uploaded file to PNG if necessary
        if (req.file.mimetype === 'image/tiff' || req.file.mimetype === 'image/tif'){
            imgBuffer = await convertTiffToPng(req.file.buffer) 
        } else if (req.file.mimetype === 'application/x-mha') {
            const pngPaths = await convertMhaToPng(req.file.buffer);
            imgBuffer = fs.readFileSync(pngPaths[0]); // Use the first slice for prediction
        } 
        else if (req.file.mimetype === 'application/dicom') {
            const pngPath = await convertDicomToPng(req.file.buffer);
            imgBuffer = fs.readFileSync(pngPath);}  
        else {
            imgBuffer = req.file.buffer; // PNG or JPEG
        }
        //const imgBuffer = await sharp(req.file.buffer).png().toBuffer();
        console.log(imgBuffer)
        const img = tf.node.decodeImage(imgBuffer, 3);
        
        // Resize the image to the required dimensions (assuming 256x256 for this example)
        const resizedImg = tf.image.resizeBilinear(img, [256, 256]);
        console.log('Image resized:', resizedImg);

        // Preprocess the image for use with the model
        const input = tf.expandDims(img.toFloat().div(255), 0);
        console.log('input::', input);

        // new  line of code 
        const input2 = tf.expandDims(resizedImg.toFloat().div(255),0);
        console.log('input::', input2);


        // Load the model
        const model = await tf.loadLayersModel(`file://${modelPath}`);
        console.log(model.outputs)
        // Run the model on the input image
        const output = model.predict(input2);
        console.log('output::', output);
        const outputData = await output.data();

        // Convert the output tensor to a binary mask
        const mask = tf.greater(output, 0.5).toInt();
        const imgg = tf.reshape(mask, [256, 256, 1]);
        const imggg = tf.mul(tf.cast(imgg, 'float32'), 255);
        const maskBuffer = await tf.node.encodePng(imggg);
        
        const uint8Array = new Uint8Array(maskBuffer);
        const base64String2 = Buffer.from(imgBuffer).toString('base64');
        const base64String = Buffer.from(uint8Array.buffer).toString('base64');

        const indexHtml = fs.readFileSync(htmlPath, 'utf-8');
        const base64Uri2 = `data:image/png;base64,${base64String2}`;
        const base64Uri = `data:image/png;base64,${base64String}`;
        const htmlWithImage = indexHtml.replace(`<img id="mask-image" />`, `<h3 id="tumor-heading">Predicted Tumor image will display here:</h3><img id="mask-image" src="${base64Uri}" /><br><h3 id="upload-heading">Uploaded Image will display here:</h3><p><img id="upload-image" src="${base64Uri2}" /></p>`);
        
        res.send(htmlWithImage);
    } catch (err) {
        console.error(err);
        res.status(500).send('An error occurred. Please check your code.');
    }
};
const handleClassificationPrediction = async (req, res, modelPath, htmlPath) => {
    try {
        console.log('it hit');
        console.log('File MIME type:', req.file.mimetype);
        // Load the image into a tensor
        let imgBuffer;

        // Convert uploaded file to PNG if necessary
        if (req.file.mimetype === 'image/tiff' || req.file.mimetype === 'image/tif'){
            imgBuffer = await convertTiffToPng(req.file.buffer) 
        }  else if (req.file.mimetype === 'application/octet-stream') { // Check file extension for .mha

            // Convert MHA to PNG using Python script
            //await convertMhaToPng(mhaPath, outputPath);
            const pngPaths = await convertMhaToPng(req.file.buffer);
            imgBuffer = fs.readFileSync(pngPaths[0]); // Use the first slice for predictionconve
            console.log('imagebuffer', imgBuffer);
        } 
        else if (req.file.mimetype === 'application/dicom') {
            const pngPath = await convertDicomToPng(req.file.buffer);
            imgBuffer = fs.readFileSync(pngPath);}  
        else {
            imgBuffer = req.file.buffer; // PNG or JPEG
        }
        //const imgBuffer = await sharp(req.file.buffer).png().toBuffer();
        //console.log(imgBuffer)
        const img = tf.node.decodeImage(imgBuffer, 3);
        
        // Resize the image to the required dimensions (assuming 256x256 for this example)
        const resizedImg = tf.image.resizeBilinear(img, [1024, 1024]);
        console.log('Image resized:', resizedImg);

        // Preprocess the image for use with the model
        const input = tf.expandDims(img.toFloat().div(255), 0);
        console.log('input::', input);

        // new  line of code 
        const input2 = tf.expandDims(resizedImg.toFloat().div(255),0);
        console.log('input::', input2);


        // Load the model
        const model = await tf.loadGraphModel(`file://${modelPath}`);
        const inputName1 = model.inputs[0].name; // Assuming the model has one input tensor
        const outputNames1 = model.outputs.map(output => output.name)
        console.log('input & output',inputName1, outputNames1)
        const inputName = 'input_tensor'; // Replace with your actual input tensor name
        const outputNames = [
            'Identity', 'Identity_1', 'Identity_2', 'Identity_3', 
            'Identity_4', 'Identity_5', 'Identity_6', 'Identity_7'
        ]; // Replace with your actual output tensor names

        // Create a dictionary for inputs
        const inputs = {};
        inputs[inputName] = input2.cast('int32');

        // Execute the model asynchronously
        const prediction = await model.executeAsync(inputs, outputNames);
        //console.log('Prediction:', prediction);

        const [boxes, scores, classes, valid_detections] = prediction;
        
        // Convert tensors to arrays
        const boxesArr = boxes.arraySync()[0];
        const scoresArr = scores.arraySync()[0];
        const classesArr = classes.arraySync()[0];
        const validDetections = valid_detections.arraySync()[0];
        console.log('boxes',boxesArr,'scoresArr',scoresArr,'classesArr',classesArr,'what detected',validDetections); 
        // Draw bounding boxes on the image
        const canvas = createCanvas(1024, 1024);
        const ctx = canvas.getContext('2d');
        const image = await loadImage(imgBuffer);
        ctx.drawImage(image, 0, 0, 1024, 1024);

        ctx.strokeStyle = 'red';
        ctx.lineWidth = 2;
        for (let i = 0; i < validDetections; i++) {
            const [y1, x1, y2, x2] = boxesArr[i];
            const score = scoresArr[i];
            const className = classesArr[i];
            ctx.strokeRect(x1 * 224, y1 * 224, (x2 - x1) * 224, (y2 - y1) * 224);
            ctx.fillText(`Class: ${className}, Score: ${score.toFixed(2)}`, x1 * 224, y1 * 224 - 10);
            console.log('ss',className,score)
        }

        const outputBuffer = canvas.toBuffer('image/png');

        const base64String2 = Buffer.from(imgBuffer).toString('base64');
        const base64String = Buffer.from(outputBuffer).toString('base64');        

        console.log('ooo',model.outputs);
        // 

        const indexHtml = fs.readFileSync(htmlPath, 'utf-8');
        const base64Uri2 = `data:image/png;base64,${base64String2}`;
        const base64Uri = `data:image/png;base64,${base64String}`;
        const htmlWithImage = indexHtml.replace(`<img id="mask-image" />`, `<h3 id="tumor-heading">Predicted Lung Nodule image will display here:</h3><img id="mask-image" src="${base64Uri}" /><br><h3 id="upload-heading">Uploaded Image will display here:</h3><p><img id="upload-image" src="${base64Uri2}" /></p>`);
        
        res.send(htmlWithImage);
    } catch (err) {
        console.error(err);
        res.status(500).send('An error occurred. Please check your code.');
    }
};
// handel bounding box predictions
const handleBoundingBoxPrediction = async (req, res, modelPath, htmlPath) => {
    try {
        console.log('it recived buffer')
        let imgBuffer;

        // Convert uploaded file to PNG if necessary
        if (req.file.mimetype === 'image/tiff' || req.file.mimetype === 'image/tif') {
            imgBuffer = await convertTiffToPng(req.file.buffer);
        } else if (req.file.mimetype === 'application/octet-stream') {
            const pngPaths = await convertMhaToPng(req.file.buffer);
            imgBuffer = fs.readFileSync(pngPaths[0]); // Use the first slice for prediction
        } else if (req.file.mimetype === 'application/dicom') {
            const pngPath = await convertDicomToPng(req.file.buffer);
            imgBuffer = fs.readFileSync(pngPath);
        } else {
            imgBuffer = req.file.buffer; // PNG or JPEG
        }

         // Prepare the input for the ONNX model
         const [input, imgWidth, imgHeight] = await prepareInput(imgBuffer);
         console.log(imgHeight,imgWidth);


         // Run the ONNX model
         const output = await runModel(input, modelPath);
         console.log(output)
 


         // Process the model output
         const boxes = processOutput(output, imgWidth, imgHeight);
         const outputImageBuffer = await drawBoundingBoxes(imgBuffer, boxes);
         //console.log('boxes', boxes)
 
         // Send the processed output to the browser
        const uint8Array = new Uint8Array(outputImageBuffer);
        const base64String2 = Buffer.from(imgBuffer).toString('base64');
        const base64String = Buffer.from(uint8Array.buffer).toString('base64');

        const indexHtml = fs.readFileSync(htmlPath, 'utf-8');
        const base64Uri2 = `data:image/png;base64,${base64String2}`;
        const base64Uri = `data:image/png;base64,${base64String}`;
        const htmlWithImage = indexHtml.replace(`<img id="mask-image" />`, `<h3 id="tumor-heading">Prediction image will display here:</h3><img id="mask-image" src="${base64Uri}" /><br><h3 id="upload-heading">Uploaded Image will display here:</h3><p><img id="upload-image" src="${base64Uri2}" /></p>`);
        
        res.send(htmlWithImage);
        //res.end(outputImageBuffer);
        //res.json(boxes);
     } catch (err) {
         console.error(err);
         res.status(500).send('An error occurred. Please check your code.');
     }
 } ;
 const drawBoundingBoxes = async (imageBuffer, boxes) => {
    const img = await loadImage(imageBuffer);
    const canvas = createCanvas(img.width, img.height);
    const ctx = canvas.getContext('2d');
    ctx.drawImage(img, 0, 0);

    ctx.strokeStyle = "#00FF00";
    ctx.lineWidth = 3;
    ctx.font = "18px serif";

    boxes.forEach(box => {
        const [x1, y1, x2, y2, label, score] = box;
        ctx.strokeRect(x1, y1, x2 - x1, y2 - y1);
        ctx.fillStyle = "#00FF00";
        const text = `${label} (${(score * 100).toFixed(2)}%)`;
        const width = ctx.measureText(text).width;
        ctx.fillRect(x1, y1 - 20, width + 10, 25);
        ctx.fillStyle = "#000000";
        ctx.fillText(text, x1, y1 - 5);
    });

    return canvas.toBuffer();
};
// Function to prepare input for the ONNX model
const prepareInput = async (buf) => {
    const img = sharp(buf);
    const md = await img.metadata();
    const [imgWidth, imgHeight] = [md.width, md.height];
    const pixels = await img.removeAlpha()
        .resize({ width: 640, height: 640, fit: 'fill' })
        .raw()
        .toBuffer();

    const red = [], green = [], blue = [];
    for (let index = 0; index < pixels.length; index += 3) {
        red.push(pixels[index] / 255.0);
        green.push(pixels[index + 1] / 255.0);
        blue.push(pixels[index + 2] / 255.0);
    }

    const input = [...red, ...green, ...blue];
    return [input, imgWidth, imgHeight];
};

// Function to run the ONNX model
const runModel = async (input, modelPath) => {
    const model = await ort.InferenceSession.create(modelPath);
    input = new ort.Tensor(Float32Array.from(input), [1, 3, 640, 640]);
    const outputs = await model.run({ images: input });
    return outputs["output0"].data;
};
const yolo_classes=[ 'boneanomaly',
    'bonelesion',
    'foreignbody',
    'fracture',
     'metal',
     'periostealreaction',
     'pronatorsign',
    'softtissue',
    'text'
]

function iou(box1,box2) {
    return intersection(box1,box2)/union(box1,box2);
}

function union(box1,box2) {
    const [box1_x1,box1_y1,box1_x2,box1_y2] = box1;
    const [box2_x1,box2_y1,box2_x2,box2_y2] = box2;
    const box1_area = (box1_x2-box1_x1)*(box1_y2-box1_y1)
    const box2_area = (box2_x2-box2_x1)*(box2_y2-box2_y1)
    return box1_area + box2_area - intersection(box1,box2)
}

function intersection(box1,box2) {
    const [box1_x1,box1_y1,box1_x2,box1_y2] = box1;
    const [box2_x1,box2_y1,box2_x2,box2_y2] = box2;
    const x1 = Math.max(box1_x1,box2_x1);
    const y1 = Math.max(box1_y1,box2_y1);
    const x2 = Math.min(box1_x2,box2_x2);
    const y2 = Math.min(box1_y2,box2_y2);
    return (x2-x1)*(y2-y1)
}


// Function to process the output of the ONNX model
function processOutput(output, img_width, img_height) {
    let boxes = [];
    for (let index=0;index<8400;index++) {
        const [class_id,prob] = [...Array(9).keys()]
            .map(col => [col, output[8400*(col+4)+index]])
            .reduce((accum, item) => item[1]>accum[1] ? item : accum,[0,0]);
        if (prob < 0.5) {
            continue;
        }
        const label = yolo_classes[class_id];
        const xc = output[index];
        const yc = output[8400+index];
        const w = output[2*8400+index];
        const h = output[3*8400+index];
        const x1 = (xc-w/2)/640*img_width;
        const y1 = (yc-h/2)/640*img_height;
        const x2 = (xc+w/2)/640*img_width;
        const y2 = (yc+h/2)/640*img_height;
        boxes.push([x1,y1,x2,y2,label,prob]);
    }

    boxes = boxes.sort((box1,box2) => box2[5]-box1[5])
    const result = [];
    while (boxes.length>0) {
        result.push(boxes[0]);
        boxes = boxes.filter(box => iou(boxes[0],box)<0.7);
    }
    return result;
}
const prepareInput1 = async (imgBuffer) => {
    // Resize the image to 224x224 and convert it to a tensor
    const image = await sharp(imgBuffer)
        .resize(224, 224)
        .raw()
        .toBuffer();

    const imgWidth = 224;
    const imgHeight = 224;

    // Normalize the image data to the range [0, 1]
    const input = Float32Array.from(image).map(pixel => pixel / 255.0);

    // Ensure the data length matches 224 * 224 * 3
    if (input.length !== imgWidth * imgHeight * 3) {
        throw new Error(`Input data length (${input.length}) does not match expected size (${imgWidth * imgHeight * 3})`);
    }

    return [input, imgWidth, imgHeight];
};



const runModel1 = async (input, modelPath) => {
    const model = await ort.InferenceSession.create(modelPath);
    inputt = new ort.Tensor(Float32Array.from(input), [1, 3, 224, 224]);
    console.log('input', input);
    const inputs = model.inputNames;
    console.log('Model inputs:', inputs);

    // Assume the first input name is the one we need
    const inputName = inputs[0];
    const outputs = await model.run({ [inputName]: inputt });
    console.log('output',outputs)
    return outputs["2053"].data
};
const processOutput1 = (output, imgWidth, imgHeight) => {
    // Assuming the model output includes probabilities for each class
    const classNames = ["FNH", "HCC", "HHE"];
    const probabilities = output;
    const expscores = probabilities.map(Math.exp);
    const sumexpscores = expscores.reduce((a,b)=>a+b,0);
    const probability = expscores.map(score => score / sumexpscores);
    return {classNames, probability };
};

const writeclassification = async (imgBuffer, boxes, classNames, probabilities) => {
    //console.log('classname');
    const svgText = boxes['classNames'].map((className, index) => `
        <text x="10" y="${30 + index * 30}" font-family="Verdana" font-size="20" fill="green">
            ${boxes['classNames']}: ${(boxes['probability'][index] * 100).toFixed(2)}%
        </text>
    `).join('');

    const svg = `
        <svg width="500" height="500" xmlns="http://www.w3.org/2000/svg">
            ${svgText}
        </svg>
    `;
    const svgBuffer = Buffer.from(svg);

    const image = await sharp(imgBuffer)
        .resize(500, 500) // Resize to ensure the SVG fits the image dimensions
        .composite([{ input: svgBuffer, top: 0, left: 0 }])
        .toBuffer();

    return image;
};
// model function for liver classification 
const handleliverclass = async (req, res, modelPath, htmlPath) => {
    try {
        console.log('it recived buffer')
        let imgBuffer;

        // Convert uploaded file to PNG if necessary
        if (req.file.mimetype === 'image/tiff' || req.file.mimetype === 'image/tif') {
            imgBuffer = await convertTiffToPng(req.file.buffer);
        } else if (req.file.mimetype === 'application/octet-stream') {
            const pngPaths = await convertMhaToPng(req.file.buffer);
            imgBuffer = fs.readFileSync(pngPaths[0]); // Use the first slice for prediction
        } else if (req.file.mimetype === 'application/dicom') {
            const pngPath = await convertDicomToPng(req.file.buffer);
            imgBuffer = fs.readFileSync(pngPath);
        } else {
            imgBuffer = req.file.buffer; // PNG or JPEG
        }

         // Prepare the input for the ONNX model
         const [input, imgWidth, imgHeight] = await prepareInput1(imgBuffer);
         console.log(imgHeight,imgWidth);


         // Run the ONNX model
         const output = await runModel1(input, modelPath);
         console.log(output)
 


         // Process the model output
         const boxes = processOutput1(output, imgWidth, imgHeight);
         console.log('boxes', boxes);
         const outputImageBuffer = await writeclassification(imgBuffer, boxes);
         console.log('boxes', outputImageBuffer)
 
         // Send the processed output to the browser
        const uint8Array = new Uint8Array(outputImageBuffer);
        const base64String2 = Buffer.from(imgBuffer).toString('base64');
        const base64String = Buffer.from(uint8Array.buffer).toString('base64');

        const indexHtml = fs.readFileSync(htmlPath, 'utf-8');
        const base64Uri2 = `data:image/png;base64,${base64String2}`;
        const base64Uri = `data:image/png;base64,${base64String}`;
        const htmlWithImage = indexHtml.replace(`<img id="mask-image" />`, `<h3 id="tumor-heading">Prediction image will display here:</h3><img id="mask-image" src="${base64Uri}" /><br><h3 id="upload-heading">Uploaded Image will display here:</h3><p><img id="upload-image" src="${base64Uri2}" /></p>`);
        
        res.send(htmlWithImage);
        //res.end(outputImageBuffer);
        //res.json(boxes);
     } catch (err) {
         console.error(err);
         res.status(500).send('An error occurred. Please check your code.');
     }
 } ;

// Routes for each model
app.get('/Tumor.html',upload.single('image') ,(req, res) => {
    //res.setHeader('Content-Type', 'text/css');
    res.sendFile(path.join(__dirname,'Tumor.html'));
});

app.post('/tumor', upload.single('image'), (req, res) => {
    console.log('Received POST request for /tumor');
    handleModelPrediction(req, res, '/tfjs2_model/model.json', path.join(__dirname, 'Tumor.html'));
});

app.get('/bone.html', (req, res) => {
    //res.setHeader('Content-Type', 'text/css');
    res.sendFile(path.join(__dirname,'bone.html'));
});
app.get('/intro.html',(req,res)=>{
    //res.setHeader('Content-Type', 'text/css');
    res.sendFile(path.join(__dirname,'intro.html'))
});

app.get('/Portfolio.html',(req,res)=>{
    //res.setHeader('Content-Type', 'text/css');
    res.sendFile(path.join(__dirname,'portfolio.html'))
});
app.post('/bone', upload.single('image'), (req, res) => {
    console.log('recived ');
    handleBoundingBoxPrediction(req, res, '/home/shivam_singh/train3_gpu/weights/best.onnx', path.join(__dirname, 'bone.html'));
    console.log('handled output');

});

app.get('/lung.html', (req, res) => {
    //res.setHeader('Content-Type', 'text/css');
    res.sendFile(path.join(__dirname, 'lung.html'));
});

app.post('/lung', upload.single('image'), (req, res) => {
    //console.log('File MIME type:', req.file.mimetype);
    handleClassificationPrediction(req, res, '/home/shivam_singh/Downloads/my_models/tfjs_model/model.json', path.join(__dirname, 'lung.html'));
    console.log('handled output');
});
app.get('/liver.html',(req,res)=>{
    res.sendFile(path.join(__dirname,'liver.html'))
});
app.post('/liver',upload.single('image'),(req,res)=>{
    handleliverclass(req, res, '/home/shivam_singh/Downloads/my_models/model.onnx', path.join(__dirname,'liver.html'))
})
// Start the server
app.listen(port, () => {
    console.log(`Server listening at http://localhost:${port}`);
});
