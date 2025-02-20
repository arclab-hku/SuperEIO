window.HELP_IMPROVE_VIDEOJS = false;

var INTERP_BASE = "./static/interpolation/stacked";
// var NUM_INTERP_FRAMES = 30; //240;

var NUM_INTERP_FRAMES
var START_INTERP_FRAMES

function initializeInterpolationPlugin(maxFrames, startIndex) {
  NUM_INTERP_FRAMES = maxFrames; // 设置最大图像数目(也就是截止到第几张图片)
  START_INTERP_FRAMES=startIndex; // 传递开始索引
}

var interp_images = [];
function preloadInterpolationImages() {
  for (var i = START_INTERP_FRAMES; i < NUM_INTERP_FRAMES; i++) {
    // var path = INTERP_BASE + '/' + String(i).padStart(6, '0') + '.jpg';
    var path = INTERP_BASE + '/图片' + (i + 1) + '.png'; // 将i + 1用于生成“图片1.png”
    interp_images[i] = new Image();
    interp_images[i].src = path;
  }
}

function setInterpolationImage(i) {
  var image = interp_images[i];
  image.ondragstart = function() { return false; };
  image.oncontextmenu = function() { return false; };
  $('#interpolation-image-wrapper').empty().append(image);
}


$(document).ready(function() {

  var maxFrames = parseInt($('#interpolation-slider').data('max-frames'));
  var startIndex = parseInt($('#interpolation-slider').data('start-index'));

   initializeInterpolationPlugin(maxFrames, startIndex); // 设置最大帧数和起始索引

    // Check for click events on the navbar burger icon
    $(".navbar-burger").click(function() {
      // Toggle the "is-active" class on both the "navbar-burger" and the "navbar-menu"
      $(".navbar-burger").toggleClass("is-active");
      $(".navbar-menu").toggleClass("is-active");

    });

    var options = {
			slidesToScroll: 1,
			slidesToShow: 3,
			loop: true,
			infinite: true,
			autoplay: false,
			autoplaySpeed: 3000,
    }

		// Initialize all div with carousel class
    var carousels = bulmaCarousel.attach('.carousel', options);

    // Loop on each carousel initialized
    for(var i = 0; i < carousels.length; i++) {
    	// Add listener to  event
    	carousels[i].on('before:show', state => {
    		console.log(state);
    	});
    }

    // Access to bulmaCarousel instance of an element
    var element = document.querySelector('#my-element');
    if (element && element.bulmaCarousel) {
    	// bulmaCarousel instance is available as element.bulmaCarousel
    	element.bulmaCarousel.on('before-show', function(state) {
    		console.log(state);
    	});
    }

    /*var player = document.getElementById('interpolation-video');
    player.addEventListener('loadedmetadata', function() {
      $('#interpolation-slider').on('input', function(event) {
        console.log(this.value, player.duration);
        player.currentTime = player.duration / 100 * this.value;
      })
    }, false);*/
    preloadInterpolationImages();

    $('#interpolation-slider').on('input', function(event) {
      setInterpolationImage(this.value);
    });
    setInterpolationImage(startIndex);
    $('#interpolation-slider').prop('min', startIndex);
    $('#interpolation-slider').prop('max', NUM_INTERP_FRAMES - 1);

    bulmaSlider.attach();

})
