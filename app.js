const tf = require("@tensorflow/tfjs-node-gpu");
const express = require("express");
const multer = require("multer");
const sharp = require("sharp");
const fs = require('fs')
const path =require("path");
const { request } = require("http");
const app = express();
const port = 3000;
const upload = multer({
    storage: multer.memoryStorage(),
    fileFilter: (req, file, cb) => {
      if (file.mimetype === "image/tiff" || "image/png" || "image/") {
        cb(null, true);
        cb(null, Date.now() + path.extname(file.originalname));

      } else {
        cb(new Error("Invalid file type. Only TIFF,png and jpeg files are allowed."));
      }
    },
  });
  
  app.get('/',(req,res)=>{
    const filepath = path.join(__dirname,"","/Tumor.html");
    const filecontent = fs.readFileSync(filepath,"utf-8"); 
    res.send(filecontent);
    //console.log('res',res)   
  });
  
  const indexHtml = fs.readFileSync(path.join(__dirname,"",'/Tumor.html'), 'utf-8');
  console.log('aboutkkkk',__dirname)
  app.use(express.static(path.join(__dirname)));
  app.use(function(req,res,next){
    //let url1 = req.url;
    //res.send("response")response
    //req.url = 'Thesis_Project/Tumor.html'
    res.header("Access-Control-Allow-Origin", "YOUR-DOMAIN.TLD"); // update to match the domain you will make the request from
    res.header("Access-Control-Allow-Headers", "Origin, X-Requested-With, Content-Type, Accept");
    console.log(`${new Date()}-${req.method} requested for give info ${req.url}`); 
    next();
  });
  // Set up route for uploading an image and running the model
  console.log('welcome')
  app.post("/", upload.single("image"), async (req, res) => {
    try {
      
      // Load the image into a tensor
      const imgBuffer = await sharp(req.file.buffer).png().toBuffer();
      console.log(imgBuffer);
     
      const img = tf.node.decodeImage(imgBuffer, 3);
      //console.log(img);
  
      // Preprocess the image for use with the model
      const input = tf.expandDims(img.toFloat().div(255), 0);
      console.log(input);
      // Load the model
      const model = await tf.loadLayersModel("file://./tfjs2_model/model.json");
     // console.log(model);
  
      // Run the model on the input image
      const output = model.predict(input);
      const outputData = await output.data();
      console.log(outputData);
     
      
            // Convert the output tensor to a binary mask
      const mask = tf.greater(output,0.5).toInt();
      console.log(mask);
     //const mask_new = tf.tensor4d(mask,[1,256,256,1]);
      const imgg=tf.reshape(mask,[256,256,1]);
      const imggg = tf.mul(tf.cast(imgg, 'float32'), 255);
      console.log(imggg);
      const maskBuffer = await tf.node.encodePng(imggg);
      console.log(mask.shape, imggg.shape);
      console.log(maskBuffer);
      //fs.writeFileSync('mask.png', maskBuffer);
      const uint8Array = new Uint8Array(maskBuffer);
   
      const base64String2=Buffer.from(imgBuffer).toString('base64');
      //console.log(base64String2);
      const base64String = Buffer.from(uint8Array.buffer).toString('base64');

     // console.log(base64String);
      console.log(maskBuffer.length);
      //console.log(typeof(maskBuffer));    
     // const maskBase645 = encodeURIComponent(maskBase64);  
     // const html = `<html><body><img src="data:image/png;base64,${base64String}}" alt="TumorIMAGE"  /></body></html>`;
      //res.send(html);
      const base64Uri2=`data:image/png;base64,${base64String2}`;
      const base64Uri = `data:image/png;base64,${base64String}`;
      //const base64Data = base64String.replace(/^data:image\/png;base64,/, '');
      //fs.writeFileSync('mask_decoded.png', Buffer.from(base64Data, 'base64'));
      //console.log(base64Uri);
      const htmlWithImage = indexHtml.replace(`<img id="mask-image" />`, `<h3 id="tumor-heading">Predicted Tumor image will display here:</h3><img id="mask-image" src="${base64Uri}" /><br><h3 id="upload-heading">Uploaded Image will display here:</h3><p><img id="upload-image" src="${base64Uri2}" /></p>`);
      res.send(htmlWithImage);
      //res.send(`<img src="data:image/png;base64,${base64String}" />`);
     // fs.writeFileSync("out.png", maskBuffer);
    } catch (err) {
      console.error(err);
      res.status(500).send("An error occurred.pls look into your code ");
    }
  });
  // Start the server
  app.listen(port, () => {
    console.log(`Server listening at http://localhost:${port}`);
    
  });
