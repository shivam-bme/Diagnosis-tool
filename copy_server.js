const express = require('express');
const multer = require('multer');
const sharp = require('sharp');
const tf = require('@tensorflow/tfjs-node');
const ort = require('onnxruntime-node');
const fs = require('fs');
const path = require('path');
const app = express();
const nifti = require('nifti-reader-js');

const dicomParser = require('dicom-parser');
const { PNG } = require('pngjs');
const port = 3000;

// Set up multer for handling file uploads
const upload = multer({
    storage: multer.memoryStorage(),
    fileFilter: (req, file, cb) => {
        if (file.mimetype === 'image/tiff' || file.mimetype === 'image/png' || file.mimetype === 'image/jpeg' || file.mimetype === 'image/jpeg' || file.mimetype === 'application/x-mha' || file.mimetype === 'application/dicom') {
            cb(null, true);
        } else {
            cb(new Error('Invalid file type. Only dicom, metal image,  TIFF, PNG, and JPEG files are allowed.'));
        }
    }
});
console.log(__dirname, "next out :",(path.join(__dirname,'/public')) );
// Serve static files
app.use(express.static(path.join(__dirname,'/public')));

// Route to serve index.html
app.get('/', (req, res) => {
    res.sendFile(path.join(__dirname, 'index.html'));
});
// conversion code 
const convertTiffToPng = async (inputBuffer) => {
    const outputBuffer = await sharp(inputBuffer).png().toBuffer();
    return outputBuffer;
};

// Function to convert MHA to PNG
const convertMhaToPng = async (inputBuffer) => {
    const outputDir = path.join(__dirname, 'converted_images');
    if (!fs.existsSync(outputDir)) {
        fs.mkdirSync(outputDir);
    }

    const { image } = await nifti.readHeader([inputBuffer])
    const imageData1 = await nifti.readImage(image, inputBuffer);
    //const imageData = itk.getMatrixJSON(image);
    const width = image.dims[1];
    const height = image.dims[2];
    const png = new PNG({ width, height, colorType: 2 });

    for (let y = 0; y < height; y++) {
        for (let x = 0; x < width; x++) {
            const idx = (width * y + x) * 4;
            const pixelValue = imageData1[idx / 4];

            png.data[idx] = pixelValue;       // Red
            png.data[idx + 1] = pixelValue;   // Green
            png.data[idx + 2] = pixelValue;   // Blue
            png.data[idx + 3] = 255;          // Alpha
        }
    }

    png.pack().pipe(fs.createWriteStream(outputPath));
    s=png.pack().pipe(fs.createWriteStream(outputPath));
    

    return pngPaths;
};
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
// handel bounding box predictions
const handleBoundingBoxPrediction = async (req, res, modelPath, htmlPath) => {
    try {
        console.log('it recived buffer')
        let imgBuffer;

        // Convert uploaded file to PNG if necessary
        if (req.file.mimetype === 'image/tiff' || req.file.mimetype === 'image/tif') {
            imgBuffer = await convertTiffToPng(req.file.buffer);
        } else if (req.file.mimetype === 'application/x-mha') {
            const pngPaths = await convertMhaToPng(req.file.buffer);
            imgBuffer = fs.readFileSync(pngPaths[0]); // Use the first slice for prediction
        } else if (req.file.mimetype === 'application/dicom') {
            const pngPath = await convertDicomToPng(req.file.buffer);
            imgBuffer = fs.readFileSync(pngPath);
        } else {
            imgBuffer = req.file.buffer; // PNG or JPEG
        }

        const img = tf.node.decodeImage(imgBuffer, 3);
        const resizedImg = tf.image.resizeBilinear(img, [640, 640]);
        console.log(resizedImg);
        const input = tf.expandDims(resizedImg.toFloat().div(255), 0);
        console.log('input_shape',input.shape);

        // Load the model
        const model = await tf.loadGraphModel(`file://${modelPath}`);
        console.log(model.outputs);
        const output = model.predict(input);
        console.log('Model Output Shape:', output.shape);
        const outputData = await output.data();
        console.log('Model Output Data:', outputData);
        
        //console.log(outputData);

        // Assume output is already in a 3-channel format suitable for displaying bounding boxes
        const imgg = tf.reshape(output, [325,112,3]);
        const imggg = tf.mul(tf.cast(imgg, 'float32'), 255);
        const bboxBuffer = await tf.node.encodePng(imggg);

        const uint8Array = new Uint8Array(bboxBuffer);
        console.log('bboxBuffer_image:', bboxBuffer);
        const base64String2 = Buffer.from(imgBuffer).toString('base64');
        const base64String = Buffer.from(uint8Array.buffer).toString('base64');

        const indexHtml = fs.readFileSync(htmlPath, 'utf-8');
        const base64Uri2 = `data:image/png;base64,${base64String2}`;
        const base64Uri = `data:image/png;base64,${base64String}`;
        const htmlWithImage = indexHtml.replace(
            `<img id="mask-image" />`,
            `<h3 id="tumor-heading">Predicted Bounding Boxes will display here:</h3><img id="mask-image" src="${base64Uri}" /><br><h3 id="upload-heading">Uploaded Image will display here:</h3><p><img id="upload-image" src="${base64Uri2}" /></p>`
        );

        res.send(htmlWithImage);
    } catch (err) {
        console.error(err);
        res.status(500).send('An error occurred. Please check your code.');
    }
} ;
// Routes for each model
app.get('/Tumor.html',upload.single('image') ,(req, res) => {
    res.sendFile(path.join(__dirname, 'Tumor.html'));
});

app.post('/tumor', upload.single('image'), (req, res) => {
    console.log('Received POST request for /tumor');
    handleModelPrediction(req, res, '/home/shivam_singh/Downloads/tfjs2_model/model.json', path.join(__dirname, 'Tumor.html'));
});

app.get('/bone.html', (req, res) => {
    res.sendFile(path.join(__dirname, 'bone.html'));
});
app.get('/intro.html',(req,res)=>{
    res.sendFile(path.join(__dirname,'intro.html'))
});

app.get('/Portfolio.html',(req,res)=>{
    res.sendFile(path.join(__dirname,'portfolio.html'))
});
app.post('/bone', upload.single('image'), (req, res) => {
    console.log('recived ');
    handleBoundingBoxPrediction(req, res, '/home/shivam_singh/model.json', path.join(__dirname, 'bone.html'));
    console.log('handled output');

});

app.get('/lung.html', (req, res) => {
    res.sendFile(path.join(__dirname, 'lung.html'));
});

app.post('/lung', upload.single('image'), (req, res) => {
    handleBoundingBoxPrediction(req, res, '/home/shivam_singh/Downloads/Bone_Fracture_Detection_YOLOv8-main/tfjs_model_last/model.json', path.join(__dirname, 'lung.html'));
    console.log('handled output');
});
// Start the server
app.listen(port, () => {
    console.log(`Server listening at http://localhost:${port}`);
});
