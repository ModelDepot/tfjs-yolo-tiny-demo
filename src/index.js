import * as tf from '@tensorflow/tfjs';
import yolo, { downloadModel } from 'tfjs-yolo-tiny';

import { Webcam } from './webcam';

let model;
const webcam = new Webcam(document.getElementById('webcam'));

(async function main() {
  try {
    ga();
    await webcam.setup();
    model = await downloadModel();

    doneLoading();
    run();
  } catch(e) {
    console.error(e);
    showError();
  }
})();

async function run() {
  clearRects();

  const inputImage = webcam.capture();

  const t0 = performance.now();

  const boxes = await yolo(inputImage, model);

  const t1 = performance.now();
  console.log("YOLO inference took " + (t1 - t0) + " milliseconds.");

  boxes.forEach(box => {
    const {
      top, left, bottom, right, classProb, className,
    } = box;

    drawRect(left, top, right-left, bottom-top,
      `${className} Confidence: ${Math.round(classProb * 100)}%`)
  });

  await tf.nextFrame();

  run();
}

const webcamElem = document.getElementById('webcam-wrapper');

function drawRect(x, y, w, h, text = '', color = 'red') {
  const rect = document.createElement('div');
  rect.classList.add('rect');
  rect.style.cssText = `top: ${y}; left: ${x}; width: ${w}; height: ${h}; border-color: ${color}`;

  const label = document.createElement('div');
  label.classList.add('label');
  label.innerText = text;
  rect.appendChild(label);

  webcamElem.appendChild(rect);
}

function clearRects() {
  const rects = document.getElementsByClassName('rect');
  while(rects[0]) {
    rects[0].parentNode.removeChild(rects[0]);
  }
}

function doneLoading() {
  const elem = document.getElementById('loading-message');
  elem.style.display = 'none';
}

function showError() {
  const elem = document.getElementById('error-message');
  elem.style.display = 'block';
  doneLoading();
}

function ga() {
  if (process.env.UA) {
    window.dataLayer = window.dataLayer || [];
    function gtag(){dataLayer.push(arguments);}
    gtag('js', new Date());
    gtag('config', process.env.UA);
  }
}
