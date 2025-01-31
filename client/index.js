function getBathValue() {
    var uiBathrooms = document.getElementsByName("uiBathrooms");
    for(var i in uiBathrooms) {
      if(uiBathrooms[i].checked) {
          return parseInt(i)+1;
      }
    }
    return -1;
  }
  
  function getBHKValue() {
    var uiBHK = document.getElementsByName("uiBHK");
    for(var i in uiBHK) {
      if(uiBHK[i].checked) {
          return parseInt(i)+1;
      }
    }
    return -1;
  }
  
  function onClickedEstimatePrice() {
    console.log("Estimate price button clicked");
    var sqft = document.getElementById("uiSqft");
    var bhk = getBHKValue();
    var bathrooms = getBathValue();
    var location = document.getElementById("uiLocations");
    var estPrice = document.getElementById("uiEstimatedPrice");

    var url = "http://127.0.0.1:5000/predict_home_price";
    console.log("Making POST request to: " + url);

    $.post(url, {
        total_sqft: parseFloat(sqft.value),
        bhk: bhk,
        bath: bathrooms,
        location: location.value
    })
    .done(function(data) {
        console.log("Response data: ", data);
        estPrice.innerHTML = "<h2>" + data.estimated_price.toString() + " Lakh</h2>";
    })
    .fail(function(xhr, status, error) {
        console.error("Error: ", status, error);
        alert("Something went wrong: " + status + " - " + error);
    });
}

  
function onPageLoad() {
  console.log("Document loaded, making GET request for locations");
  var url = "http://127.0.0.1:5000/get_location_names";

  $.get(url)
  .done(function(data) {
      console.log("Response data: ", data);
      if(data) {
          var locations = data.locations;
          var uiLocations = document.getElementById("uiLocations");
          $('#uiLocations').empty();
          for (var i in locations) {
              var opt = new Option(locations[i], locations[i]);
              $('#uiLocations').append(opt);
          }
      }
  })
  .fail(function(xhr, status, error) {
      console.error("Error: ", status, error);
      alert("Error fetching locations: " + status + " - " + error);
  });
}

  
  window.onload = onPageLoad;