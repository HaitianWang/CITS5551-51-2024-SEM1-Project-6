<template>
  <div class="main-page">
    <div class="field-section">
      <img src="@/assets/1.png" alt="image" class="field-image">
      <div class="field-text">
        <img src="@/assets/3.png" alt="icon" class="small-icon"> <!-- 小图片的位置 -->
        <h2>Harnessing Multispectral Imaging and Deep
          Learning for Precision Agriculture:<br/><br/>
          The UWAIntelliCrop Initiative</h2>
      </div>
    </div>
    <div class="intro-text">
      <p>
        The primary objective of this project is to conceive, develop, and implement the application system
        named UWAIntelliCrop, a robust and integrated system that harnesses the power of advanced
        multispectral imaging combined with sophisticated deep learning techniques. This system will be
        tailored to accurately differentiate between various crop types and weed species, enabling precise
        monitoring and assessment of the extent and severity of crop damage. By fusing these insights
        with GIS data, UWAIntelliCrop will not only visualize the impacts on a macro scale but also
        generate detailed analytical reports and actionable recommendations for effective farm
        management. The ultimate aim is to support farmers in the Kondinin region—and potentially
        beyond—in implementing efficacious management strategies to minimize economic losses.
        Through its deployment on cloud infrastructure, UWAIntelliCrop promises accessibility and
        utility to users across diverse locations, offering a scalable solution in the domain of precision
        agriculture.
      </p>
    </div>
  </div>
</template>

<script>
  export default {
    name: "mainPage",
    created() {
      this.$emit('header', true);
      this.$emit('footer', true);
    },
  };
</script>

<style scoped>
  .main-page {
    text-align: center;
    padding: 20px;
  }

  .field-section {
    display: flex;
    justify-content: center;
    align-items: center;
    margin-bottom: 20px;
  }

  .field-image {
    width: 60%; /* Adjust the width as needed */
    height: auto;
    margin-right: 20px;
  }

  .field-text {
    max-width: 30%;
    text-align: left;
  }

  .field-text .small-icon {
    display: block;
    margin-bottom: 20px; /* Adjust spacing as needed */
    width: 100%; /* Adjust the width as needed */
    height: auto;
  }

  .field-text h2 {
    font-size: 1.5em;
    color: #333;
  }

  .intro-text {
    margin: 30px;
    font-size: 1.2em;
    line-height: 1.6em;
    color: #333;
    padding: 10px;
    background-color: #f9f9f9;
    border-radius: 8px;
    text-align: left;
  }
</style>
