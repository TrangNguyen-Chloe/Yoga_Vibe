// script for learn mode on web flask app

Webcam.set({
    width: 640,
    height: 480,
    image_format: 'jpeg',
    jpeg_quality: 90
});
Webcam.attach( '#my_camera' );

var current_pose_detect="";

function take_snapshot() {
    // take snapshot and get image data
    pose_webcam_dectect();
}


function pose_webcam_dectect(){
    Webcam.snap( function(data_uri) {
        // display results in page

        // document.getElementById('results').innerHTML = 
        //     '<h2>Here is your image:</h2>' + 
        //     '<img src="'+data_uri+'"/>';
        
        var image_base64 = data_uri;
        let json_data = {'data-uri': image_base64};
        console.log("Sending frame to flask")
        fetch('/send_image', {
            method: 'POST',
            headers: {
            'Accept': 'application/json, text/plain, */*',
            'Content-Type': 'application/json; charset=utf-8'
            },
            body: JSON.stringify(json_data)
        })
        .then(res=>res.json())
        .then(res => {
            console.log("Frame Detection:",res.label);
            current_pose_detect = res.label;
        }).catch(err=>console.log(err))
    } );
}


webcam_interval = setInterval(pose_webcam_dectect, 2000);

document.addEventListener("DOMContentLoaded", function(event) {

    
    function timer(){
        var pose_list = ['mountain', 'warrior1', 'warrior2', 'downdog', 'goddess', 'tree']
        var pose_img_link = ['\\static\\images\\mountain.png', 
                            '\\static\\images\\warrior1.png', 
                            '\\static\\images\\warrior2.png', 
                            '\\static\\images\\downdog.png', 
                            '\\static\\images\\goddess.png', 
                            '\\static\\images\\tree.png']


        var pose_dom = document.getElementById("pose");
        var img_dom = document.getElementById("poseImg");
        var time_dom = document.getElementById("time");
        var compliment_dom = document.getElementById("compliment")
        
        var time_total = 0;
        var errors = 0; 
        var index=0;
        var time_each = 10
        var timeleft = time_each;

        pose_dom.innerHTML=pose_list[index];
        img_dom.src=pose_img_link[index];

        var countdown_timer = setInterval(function(){
            if(timeleft <= 0){
                
                if (time_total >= pose_list.length*time_each){
                    clearInterval(countdown_timer);
                    clearInterval(webcam_interval);
                    compliment_dom.innerHTML="Marvelous! You have finished!✨"
                } else {
                    index+=1;
                    pose_dom.innerHTML=pose_list[index];
                    img_dom.src=pose_img_link[index];
                    timeleft = time_each;
                }
                
            }
            document.getElementById("progressBar").value = time_each - timeleft;
            time_dom.innerText = timeleft;

            console.log("Time left:", timeleft);
            console.log("Index:", index);
            console.log("Time total:", time_total);
            console.log("Pose Detection", current_pose_detect);

            if (current_pose_detect != pose_list[index]){
                errors += 1;
                if (errors == 3){
                    timeleft = time_each;
                    time_total = index*time_each;
                    errors = 0;
                }
            } else{
                timeleft -= 1;
                time_total += 1;
            }
        }, 1000);
    }

    timer();

});






