
function onCapture(){
    console.log('yes')
    var name = document.getElementById('name').value;
    var id = document.getElementById('id').value;
    var total_id = name
    total_id += '_'
    total_id += id
    console.log(total_id)
}

function onRecognize(){
    console.log('called')
}
