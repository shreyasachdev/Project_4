$(document).ready(function() {
    // all custom jQuery will go here
    console.log("this works")

	let ChosenImg;
	$("#image-selector").change(function() {
		let reader = new FileReader();
		reader.onload = function(e) {
			let dataURL = reader.result;
			$("#selected-image").attr("src", dataURL);
			ChosenImg = dataURL.replace("data:image/jpeg;base64,","")
			console.log(ChosenImg);
		}
		reader.readAsDataURL($("#image-selector")[0].files[0]);
		$("#prediction").text("");

	$("#predict-button").click(function(event){
		let message = {
			image : ChosenImg
		}
		
        $.post("http://127.0.0.1:5000/predict", JSON.stringify(message), 
        function(response){
            $("prediction").text(response.prediction.tofixed(3));
            console.log(response);
        	});
    	});
	});
});