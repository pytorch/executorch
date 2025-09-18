/**
 * Copyright (c) Meta Platforms, Inc. and affiliates.
 * All rights reserved.
 *
 * This source code is licensed under the BSD-style license found in the
 * LICENSE file in the root directory of this source tree.
 */

document.addEventListener("DOMContentLoaded", function() {
  const steps = Array.from(document.querySelectorAll('.progress-bar-item'));
  const h2s = Array.from(document.querySelectorAll('h2'));

  // Populate captions from h2s
  h2s.forEach((h2, index) => {
    const captionElem = document.getElementById(`caption-${index + 1}`);
    if (captionElem) {
      captionElem.innerText = h2.innerText;
    }
  });

  // Throttle function to optimize performance
  function throttle(func, delay) {
    let lastCall = 0;
    return function() {
      const now = Date.now();
      if (now - lastCall < delay) return;
      lastCall = now;
      func.apply(this, arguments);
    }
  }

  document.addEventListener("scroll", throttle(function() {
    let activeIndex = 0;
    let closestDistance = Number.MAX_VALUE;
    const totalHeight = document.documentElement.scrollHeight;
    const viewportHeight = window.innerHeight;
    const scrollBottom = window.scrollY + viewportHeight;
    const isAtBottom = totalHeight === scrollBottom;

    h2s.forEach((h2, index) => {
      const rect = h2.getBoundingClientRect();
      const distanceToTop = Math.abs(rect.top);
      if (distanceToTop < closestDistance) {
        closestDistance = distanceToTop;
        activeIndex = index;
      }
    });

    steps.forEach((step, index) => {
      if (isAtBottom) {
        step.classList.remove('active');
        step.classList.add('completed');
      } else {
        if (index < activeIndex) {
          step.classList.remove('active');
          step.classList.add('completed');
        } else if (index === activeIndex) {
          step.classList.add('active');
          step.classList.remove('completed');
        } else {
          step.classList.remove('active', 'completed');
        }
      }
    });
  }, 100));
});
