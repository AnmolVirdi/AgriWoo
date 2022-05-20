const express = require('express');

const app = express();

const path = require("path");
app.use(express.static(path.join(__dirname, "public")));
app.use(express.json({limit: '10mb'}));
app.use(express.urlencoded({extended: true, limit: '10mb'}));
const fs = require('fs');
app.get("/", (req, res) => {
    res.sendFile("index.html");
});
  
app.get("/login", (req, res) => {
    res.sendFile(__dirname+"/public/login.html");
});
  
app.get("/upload", (req, res) => {
    res.sendFile(__dirname + "/public/upload.html");
});

app.post('/image-data', (req,res)=>{
    const {base64} = req.body;
    fs.writeFile("anything.jpg", base64, 'base64', function(err){
        if (err) console.log(err);
    })

    res.json({msg: 'File Succesfully uploaded'});
})


app.listen(3001, ()=>console.log('Server up and running on port 3001'));