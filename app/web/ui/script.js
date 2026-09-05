document.addEventListener('DOMContentLoaded', () => {
  const userInput = document.getElementById('user-input');
  const systemOutput = document.getElementById('system-output');
  const generateBtn = document.getElementById('generate-btn');

  let generationCount = 0;

  function generateSystemResponse(inputValue) {
    generationCount++;
    const timestamp = new Date().toLocaleTimeString('vi-VN');
    const trimmedInput = inputValue.trim();

    if (trimmedInput) {
      return "";
    } else {
      return "";
    }
  }

  if (generateBtn) {
    generateBtn.addEventListener('click', () => {
      const generatedContent = generateSystemResponse(userInput.value);
      systemOutput.value = generatedContent;
    });
  }

  if (systemOutput) {
    systemOutput.addEventListener('click', () => {
      const currentContent = systemOutput.value || '';
    });

    systemOutput.addEventListener('keydown', (event) => {
      if (event.ctrlKey || event.metaKey || ['Tab', 'ArrowUp', 'ArrowDown', 'ArrowLeft', 'ArrowRight'].includes(event.key)) {
        return;
      }
      event.preventDefault();
    });
  }


  if (userInput) {
    userInput.addEventListener('keydown', (event) => {
      if ((event.ctrlKey || event.metaKey) && event.key === 'Enter') {
        event.preventDefault();
        generateBtn.click();
      }
    });
  }
});
